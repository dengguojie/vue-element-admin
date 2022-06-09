/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file concat_d.cpp
 * \brief
 */
#include "concat_d.h"

#include <nlohmann/json.hpp>
#include <sstream>
#include <cctype>
#include "error_log.h"
#include "vector_tiling_profiling.h"
#include "register/op_impl_registry.h"
#include "op_tiling_util.h"
#include "runtime2_util.h"
#include "op_util.h"

namespace optiling {
constexpr size_t EXPAND_HW_DIM_NUM = 2;
constexpr int64_t FZSHAPE_INDEX_C1 = 0;
constexpr int64_t FZSHAPE_INDEX_H = 1;
constexpr int64_t FZSHAPE_INDEX_W = 2;
constexpr size_t EXPAND_DHW_DIM_NUM = 3;
constexpr int64_t FZDSHAPE_INDEX_D = 0;
constexpr int64_t FZDSHAPE_INDEX_C1 = 1;
constexpr int64_t FZDSHAPE_INDEX_H = 2;
constexpr int64_t FZDSHAPE_INDEX_W = 3;

struct FzdShapePos {
  int32_t input_num;
  int32_t pos_c;
  int32_t pos_h;
  int32_t pos_w;
  int32_t pos_d;
};

static void GetNewFZShape(vector<gert::Shape>& input_shapes, const vector<gert::Shape>& input_origin_shapes,
                          int64_t c0_size, const FzdShapePos& pos) {
  size_t dim_num = input_shapes[0].GetDimNum() + EXPAND_HW_DIM_NUM;

  for (int32_t i = 0; i < pos.input_num; i++) {
    input_shapes[i].SetDimNum(dim_num);
    for (size_t j = dim_num - 1; j > EXPAND_HW_DIM_NUM; j--) {
      input_shapes[i].SetDim(j, input_shapes[i][j-EXPAND_HW_DIM_NUM]);
    }
    int64_t dim_c1 = (c0_size == 0) ? input_origin_shapes[i][pos.pos_c] : input_origin_shapes[i][pos.pos_c] / c0_size;
    input_shapes[i].SetDim(FZSHAPE_INDEX_C1, dim_c1);
    input_shapes[i].SetDim(FZSHAPE_INDEX_H, input_origin_shapes[i][pos.pos_h]);
    input_shapes[i].SetDim(FZSHAPE_INDEX_W, input_origin_shapes[i][pos.pos_w]);
  }
}

static void GetNewFZDShape(vector<gert::Shape>& input_shapes, const vector<gert::Shape>& input_origin_shapes,
                           int64_t c0_size, const FzdShapePos& pos) {
  size_t dim_num = input_shapes[0].GetDimNum() + EXPAND_DHW_DIM_NUM;

  for (int32_t i = 0; i < pos.input_num; i++) {
    input_shapes[i].SetDimNum(dim_num);
    for (size_t j = dim_num - 1; j > EXPAND_DHW_DIM_NUM; j--) {
      input_shapes[i].SetDim(j, input_shapes[i][j-EXPAND_DHW_DIM_NUM]);
    }
    input_shapes[i].SetDim(FZDSHAPE_INDEX_D, input_origin_shapes[i][pos.pos_d]);
    int64_t dim_c1 = (c0_size == 0) ? input_origin_shapes[i][pos.pos_c] : input_origin_shapes[i][pos.pos_c] / c0_size;
    input_shapes[i].SetDim(FZDSHAPE_INDEX_C1, dim_c1);
    input_shapes[i].SetDim(FZDSHAPE_INDEX_H, input_origin_shapes[i][pos.pos_h]);
    input_shapes[i].SetDim(FZDSHAPE_INDEX_W, input_origin_shapes[i][pos.pos_w]);
  }
}

static bool AdjustShape(gert::TilingContext *context, vector<gert::Shape>& input_shapes,
                        const vector<gert::Shape>& input_origin_shapes, const int32_t input_num,
                        const ge::Format& in_format, const ge::Format& ori_format) {
  constexpr int32_t pos_c0 = 3;
  constexpr size_t dim_size = 4;
  FzdShapePos pos{input_num, 0, 0, 0, 0};
  int64_t c0_size = 1;

  // dimensions of fractal_z, fractal_z_3d should be 4
  if (input_shapes[0].GetDimNum() != dim_size) {
    return true;
  } else {
    c0_size = input_shapes[0][pos_c0];
    OP_TILING_CHECK(c0_size == 0, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "c0 cannot be 0"),
                    return false);
  }

  std::string origin_format = to_string(ori_format);
  pos.pos_c = std::strchr(origin_format.c_str(), 'C') - origin_format.c_str();
  pos.pos_h = std::strchr(origin_format.c_str(), 'H') - origin_format.c_str();
  pos.pos_w = std::strchr(origin_format.c_str(), 'W') - origin_format.c_str();
  // for '1' in origin format, such as NC1HWC0
  if (std::strchr(origin_format.c_str(), '1') != NULL) {
    pos.pos_h = pos.pos_h - 1;
    pos.pos_w = pos.pos_w - 1;
    c0_size = 1;
  }
  OP_TILING_CHECK(pos.pos_c < 0, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get c position failed"),
                  return false);
  OP_TILING_CHECK(pos.pos_h < 0, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get h position failed"),
                  return false);
  OP_TILING_CHECK(pos.pos_w < 0, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get w position failed"),
                  return false);
  
  if (in_format == FORMAT_FRACTAL_Z_3D) {
    pos.pos_d = std::strchr(origin_format.c_str(), 'D') - origin_format.c_str();
    OP_TILING_CHECK(pos.pos_d < 0, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get d position failed"),
                    return false);
    GetNewFZDShape(input_shapes, input_origin_shapes, c0_size, pos);
  } else {
    GetNewFZShape(input_shapes, input_origin_shapes, c0_size, pos);
  }

  OP_LOGD(context->GetNodeName(),
          "origin format is [%s], the c, w, h, d position is: [%d, %d, %d, %d].",
          origin_format.c_str(), pos.pos_c, pos.pos_w, pos.pos_h, pos.pos_d);

  return true;
}

vector<gert::Shape> GetInputOriginShapes(gert::TilingContext *context, size_t ir_index) {
  vector<gert::Shape> shapes;
  for (size_t i = 0; ; i++) {
    auto src_storage_shape = context->GetDynamicInputShape(ir_index, i);
    if (nullptr == src_storage_shape) return shapes;
    shapes.push_back(src_storage_shape->GetOriginShape());
  }
  return shapes;
}

vector<gert::Shape> GetInputShapes(gert::TilingContext *context, size_t ir_index) {
  vector<gert::Shape> shapes;
  for (size_t i = 0; ; i++) {
    auto src_storage_shape = context->GetDynamicInputShape(ir_index, i);
    if (nullptr == src_storage_shape) return shapes;
    shapes.push_back(src_storage_shape->GetStorageShape());
  }
  return shapes;
}


static int32_t GetDimensionSize(const vector<gert::Shape>& shapes) {
  int32_t dim_size = 0;

  for (size_t i = 0; i < shapes.size(); i++) {
    if (shapes[i].GetDimNum() > 0) {
      dim_size = static_cast<int32_t>(shapes[i].GetDimNum());
      break;
    }
  }

  return dim_size;
}

bool ConcatDSLTiling(gert::TilingContext *context, const ConcatDCompileInfo *compile_info) {
  vector<gert::Shape> input_shapes = GetInputShapes(context, 0);
  OP_TILING_CHECK(
    input_shapes.empty(),
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get input_shapes failed."),
    return ge::GRAPH_FAILED);

  // get attr for concat_dim
  int32_t axis = compile_info->ori_axis;
  // get attr for N
  int32_t input_num = input_shapes.size();

  auto src_td = context->GetInputDesc(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, src_td);
  ge::Format in_format = src_td->GetStorageFormat();
  ge::Format ori_format = src_td->GetOriginFormat();
  DataType data_type = src_td->GetDataType();
  OP_LOGD(context->GetNodeName(), "input format is [%s], origin format is [%s].",
          to_string(in_format).c_str(), to_string(ori_format).c_str());
  // only adjust shape for fractal_z, fractal_z_3d
  if (in_format == FORMAT_FRACTAL_Z || in_format == FORMAT_FRACTAL_Z_3D) {
    vector<gert::Shape> input_origin_shapes = GetInputOriginShapes(context, 0);
    OP_TILING_CHECK(!AdjustShape(context, input_shapes, input_origin_shapes, input_num, in_format, ori_format),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "adjust shape failed"),
                    return ge::GRAPH_FAILED);
  }

  // the ConcatDslTilingHandler need the axis must be > 0, so need do remaining in op
  if (axis < 0) {
    axis = axis + GetDimensionSize(input_shapes);
  }

  vector<int64_t> axis_list = {axis};
  size_t dim_num = input_shapes[0].GetDimNum();
  vector<gert::Shape> dim_shapes(input_num, gert::Shape());
  for (int32_t i = 0; i < input_num; i++) {
    for (size_t j = 0; j < dim_num; j++) {
      dim_shapes[i].AppendDim(input_shapes[i].GetDim(j));
    }
  }

  // do ConcatDslTilingHandler ->DoTiling
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info->dsl_compile_info);
  OpInfo concat_info(compile_info->dsl_compile_info.get());
  concat_info.SetInputShape(&dim_shapes);
  concat_info.SetInputType(&data_type);
  concat_info.SetAxes(&axis_list);
  OP_TILING_CHECK(!DoAutoTiling(context, &concat_info),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "call DoTiling failed"),
                  return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

static bool CheckParams(gert::TilingContext *context, const vector<gert::Shape>& input_shapes, int32_t dim) {
  if (input_shapes.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "The input count should be more than 0");
    return false;
  }

  auto shape_length = input_shapes[0].GetDimNum();
  auto not_equal_shape_length = [shape_length](const gert::Shape input_shape) {
      return input_shape.GetDimNum() != shape_length;
  };
  if (std::any_of(input_shapes.begin(), input_shapes.end(), not_equal_shape_length)) {
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "The length of each shape must be equal");
      return false;
  }

  int32_t max_dim = static_cast<int32_t>(shape_length);
  if (dim >= max_dim or dim < -max_dim) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        context->GetNodeName(), "the parameter[%s] should be [between %d and %d], but actually is [%d].",
        "concat_dim", min(max_dim - 1, -max_dim), max(max_dim - 1, -max_dim), dim);
    return false;
  }

  size_t new_dims = dim >= 0 ? dim : max_dim + dim;
  const auto& shape = input_shapes[0];
  for (size_t i = 1; i < input_shapes.size(); i++) {
    for (size_t j = 0; j < shape_length; j++) {
      if (j != new_dims && input_shapes[i][j] != shape[j]) {
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                        "dims must equal except concat dim[%d], input_values[%s]",
                                        dim, ops::ToString(input_shapes).c_str());
        return false;
      }
    }
  }

  return true;
}

static bool GetTilingParam(const vector<gert::Shape>& input_shapes,
                           int32_t concat_dim, ConcatDTilingData *tilingdata) {
  if (input_shapes.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING("concat", "The input count should be more than 0");
    return false;
  }

  if (concat_dim < 0) {
    concat_dim += input_shapes[0].GetDimNum();
  }

  auto input_count = input_shapes.size();
  tilingdata->max_inner_dims = 0;
  tilingdata->axis = 1;
  tilingdata->input_count = input_shapes.size();
  tilingdata->out_dims = GetPartShapeSize(input_shapes[0], 0, concat_dim);
  tilingdata->min_inner_dims = GetPartShapeSize(input_shapes[0], concat_dim, input_shapes[0].GetDimNum());
  int64_t output_index = 0;
  for (size_t i = 0; i < input_count; i++) {
    auto inner_dims = GetPartShapeSize(input_shapes[i], concat_dim, input_shapes[0].GetDimNum());
    tilingdata->max_inner_dims = max(tilingdata->max_inner_dims, inner_dims);
    tilingdata->min_inner_dims = min(tilingdata->min_inner_dims, inner_dims);

    tilingdata->input_info[i].inner_dims = inner_dims;
    tilingdata->input_info[i].output_index = output_index;

    output_index += inner_dims;
  }

  tilingdata->output_inner_length = output_index;

  return true;
}

ge::graphStatus ConcatTIKTiling(gert::TilingContext *context, const ConcatDCompileInfo *compile_info) {
  int32_t concat_dim = compile_info->ori_axis;
  size_t input_size = static_cast<size_t>(compile_info->input_size);
  int32_t block_dim = compile_info->core_num;
  vector<gert::Shape> input_shapes = GetInputShapes(context, 0);
  OP_TILING_CHECK(
    input_shapes.size() != input_size,
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get input_shapes failed."),
    return ge::GRAPH_FAILED);

  if (!CheckParams(context, input_shapes, concat_dim)) {
    return ge::GRAPH_FAILED;
  }

  auto tilingdata = context->GetTilingData<ConcatDTilingData>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, tilingdata);
  if (!GetTilingParam(input_shapes, concat_dim, tilingdata)) {
    return ge::GRAPH_FAILED;
  }

  // block_dim, not need for concat tiling
  context->SetBlockDim(block_dim);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConcatDTiling(gert::TilingContext *context) {
  auto compile_info = reinterpret_cast<const ConcatDCompileInfo *>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  if (compile_info->is_tik) {
    return ConcatTIKTiling(context, compile_info);
  }

  return ConcatDSLTiling(context, compile_info);
}

ge::graphStatus ConcatDParse(gert::TilingParseContext *context) {
  auto compile_info = MutableCompileInfo<ConcatDCompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetJsonObj(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);
  if (GetCompileValue(*parsed_object_cinfo, "is_tik", compile_info->is_tik)) {
    const nlohmann::json& vars = (*parsed_object_cinfo)["vars"];
    OP_TILING_CHECK(!GetCompileValue(vars, "block_dim", compile_info->core_num),
                    VECTOR_INNER_ERR_REPORT_TILIING("ConcatD", "ConcatParseFunc, get block_dim error"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!GetCompileValue(vars, "concat_dim", compile_info->ori_axis),
                    VECTOR_INNER_ERR_REPORT_TILIING("ConcatD", "ConcatParseFunc, get concat_dim error"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!GetCompileValue(vars, "input_size", compile_info->input_size),
                    VECTOR_INNER_ERR_REPORT_TILIING("ConcatD", "ConcatParseFunc, get input_size error"),
                    return ge::GRAPH_FAILED);
  } else {
    compile_info->dsl_compile_info = ParseAutoTiling("ConcatD", *parsed_object_cinfo);
    OP_TILING_CHECK(compile_info->dsl_compile_info == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING("ConcatD", "CreateAutoTilingHandler return nullptr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!GetCompileValue(*parsed_object_cinfo, "concat_dim", compile_info->ori_axis),
                    VECTOR_INNER_ERR_REPORT_TILIING("ConcatD", "ConcatParseFunc, get concat_dim error"),
                    return ge::GRAPH_FAILED);
    compile_info->is_tik = false;
  }

  return ge::GRAPH_SUCCESS;
}

// register tiling interface of the ConcatD op.
IMPL_OP(ConcatD).Tiling(ConcatDTiling).TilingParse<ConcatDCompileInfo>(ConcatDParse);
}  // namespace optiling
