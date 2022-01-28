/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
 * \file concat.cpp
 * \brief dynamic shape tiling of concat
 */
#include <string>

#include "op_tiling_util.h"
#include "op_tiling/tiling_handler.h"

#include "op_log.h"
#include "../op_proto/util/error_util.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
using namespace ge;
using namespace std;

// define ignore_idx attr idx and name
static const std::pair<int64_t, std::string> AXIS_ATTR_INFO{0, "axis"};
static const int32_t AXIS_DEFAULT_VALUE = 0;

struct ConcatCompileInfo {
  std::shared_ptr<AutoTilingHandler> outer_compile_info;
  int32_t ori_axis{-1};
  int32_t core_num{1};
  int32_t input_size{1};
  bool is_tik{false};
};

static void Pack2ConcatParams(vector<int64_t>& input_shape, int32_t& concat_dim) {
  if (concat_dim < -1) {
    concat_dim += 1;
    return;
  }

  int32_t shape_length = static_cast<int32_t>(input_shape.size());
  if (concat_dim == -1 || concat_dim == shape_length) {
    input_shape.push_back(1);
  }
}

static void GetNewFZShape(vector<vector<int64_t>>& input_shapes, const vector<vector<int64_t>>& input_origin_shapes,
                          int64_t axis_c0, const int32_t input_num, const int32_t pos_c, const int32_t pos_h,
                          const int32_t pos_w) {
  int64_t axis_c1;
  int64_t axis_h;
  int64_t axis_w;

  // to avoid warning
  if (axis_c0 == 0) {
    axis_c0 = 1;
  }

  for (int32_t i = 0; i < input_num; i++) {
    axis_c1 = input_origin_shapes[i][pos_c] / axis_c0;
    axis_h = input_origin_shapes[i][pos_h];
    axis_w = input_origin_shapes[i][pos_w];
    vector<int64_t> c1hw_dims{axis_c1, axis_h, axis_w};
    input_shapes[i].erase(input_shapes[i].begin());
    input_shapes[i].insert(input_shapes[i].begin(), c1hw_dims.begin(), c1hw_dims.end());
  }
}

static void GetNewFZDShape(vector<vector<int64_t>>& input_shapes, const vector<vector<int64_t>>& input_origin_shapes,
                           int64_t axis_c0, const int32_t input_num, const int32_t pos_c, const int32_t pos_h,
                           const int32_t pos_w, const int32_t pos_d) {
  int64_t axis_c1;
  int64_t axis_h;
  int64_t axis_w;
  int64_t axis_d;

  // to avoid warning
  if (axis_c0 == 0) {
    axis_c0 = 1;
  }

  for (int32_t i = 0; i < input_num; i++) {
    axis_d = input_origin_shapes[i][pos_d];
    axis_c1 = input_origin_shapes[i][pos_c] / axis_c0;
    axis_h = input_origin_shapes[i][pos_h];
    axis_w = input_origin_shapes[i][pos_w];
    vector<int64_t> dc1hw_dims{axis_d, axis_c1, axis_h, axis_w};
    input_shapes[i].erase(input_shapes[i].begin());
    input_shapes[i].insert(input_shapes[i].begin(), dc1hw_dims.begin(), dc1hw_dims.end());
  }
}

static bool AdjustShape(const std::string& op_type, vector<vector<int64_t>>& input_shapes,
                        const vector<vector<int64_t>>& input_origin_shapes, const int32_t input_num,
                        const ge::Format& in_format, const ge::Format& ori_format) {
  int32_t pos_h = 0;
  int32_t pos_w = 0;
  int32_t pos_c = 0;
  int32_t pos_d = 0;
  int32_t pos_c0 = 3;
  int64_t axis_c0 = 1;
  size_t dim_size = 4;

  // dimensions of fractal_z, fractal_z_3d should be 4
  if (input_shapes[0].size() != dim_size) {
    return true;
  } else {
    axis_c0 = input_shapes[0][pos_c0];
    OP_TILING_CHECK(axis_c0 == 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "c0 cannot be 0"), return false);
  }

  std::string origin_format = to_string(ori_format);
  pos_c = std::strchr(origin_format.c_str(), 'C') - origin_format.c_str();
  pos_h = std::strchr(origin_format.c_str(), 'H') - origin_format.c_str();
  pos_w = std::strchr(origin_format.c_str(), 'W') - origin_format.c_str();
  // for '1' in origin format, such as NC1HWC0
  if (std::strchr(origin_format.c_str(), '1') != NULL) {
    pos_h = pos_h - 1;
    pos_w = pos_w - 1;
    axis_c0 = 1;
  }
  OP_TILING_CHECK(pos_c < 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get c position failed"), return false);
  OP_TILING_CHECK(pos_h < 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get h position failed"), return false);
  OP_TILING_CHECK(pos_w < 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get w position failed"), return false);
  
  if (in_format == FORMAT_FRACTAL_Z_3D) {
    pos_d = std::strchr(origin_format.c_str(), 'D') - origin_format.c_str();
    OP_TILING_CHECK(pos_d < 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get d position failed"), return false);
    GetNewFZDShape(input_shapes, input_origin_shapes, axis_c0, input_num, pos_c, pos_h, pos_w, pos_d);
  } else {
    GetNewFZShape(input_shapes, input_origin_shapes, axis_c0, input_num, pos_c, pos_h, pos_w);
  }

  OP_LOGD(op_type,
          "origin format is [%s], the c, w, h, d position is: [%d, %d, %d, %d].",
          origin_format.c_str(), pos_c, pos_w, pos_h, pos_d);

  return true;
}

bool ConcatParseFunc(const std::string& op_type, const nlohmann::json& compile_info, ConcatCompileInfo& op_info) {
  if (GetCompileValue(compile_info, "is_tik", op_info.is_tik)) {
    const nlohmann::json& all_vars = compile_info["vars"];
    OP_TILING_CHECK(!GetCompileValue(all_vars, "block_dim", op_info.core_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ConcatParseFunc, get block_dim error"),
                  return false);
    OP_TILING_CHECK(!GetCompileValue(all_vars, "concat_dim", op_info.ori_axis),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ConcatParseFunc, get concat_dim error"),
                  return false);
    OP_TILING_CHECK(!GetCompileValue(all_vars, "input_size", op_info.input_size),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ConcatParseFunc, get input_size error"),
                  return false);
  } else {
    op_info.outer_compile_info = CreateConcatDslTilingHandler(op_type, "Concat", compile_info);
    OP_TILING_CHECK(!GetCompileValue(compile_info, "_ori_axis", op_info.ori_axis),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ConcatParseFunc, get concat_dim error"),
                  return false);
    op_info.is_tik = false;
  }

  return true;
}

vector<vector<int64_t>> GetInputOriginShapes(const ge::Operator& paras) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(paras);
  if (op_desc == nullptr)
    return {};

  vector<vector<int64_t>> shapes;
  int count = op_desc->GetInputsSize();
  for (int i = 0; i < count; i++) {
    auto ptr = op_desc->MutableInputDesc(i);
    shapes.emplace_back(ptr->GetOriginShape().GetDims());
  }

  return shapes;
}

static int32_t GetDimensionSize(const vector<vector<int64_t>>& shapes) {
  int32_t dim_size = 0;

  for (size_t i = 0; i < shapes.size(); i++) {
    if (!shapes[i].empty()) {
      dim_size = static_cast<int32_t>(shapes[i].size());
      break;
    }
  }

  return dim_size;
}

bool ConcatDSLTiling(const std::string& op_type, const ge::Operator& op_paras, const ConcatCompileInfo& op_info,
                     utils::OpRunInfo& run_info) {
  OP_LOGD(op_type, "ConcatDSLTiling running.");
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  auto input_desc = operator_info->MutableInputDesc(0);
  vector<vector<int64_t>> input_shapes = GetInputShapes(op_paras);
  if (input_shapes.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input shapes failed.");
    return false;
  }

  int32_t axis = 0;
  // get attr for N
  int32_t input_num = input_shapes.size();
  if (op_type == "Pack") {
    OP_TILING_CHECK(input_num == 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "attr N must be > 0"), return false);
    // get attr for axis
    ops::GetAttrValue(op_paras, AXIS_ATTR_INFO, axis, AXIS_DEFAULT_VALUE);

    ScalarToShape(input_shapes[0]);
    Pack2ConcatParams(input_shapes[0], axis);
    input_shapes.assign(input_num, input_shapes[0]);
  } else {
    if (input_num == 0) {
      input_num = 1;
    }
    // get attr for concat_dim
    axis = op_info.ori_axis;

    ge::Format in_format = input_desc->GetFormat();
    ge::Format ori_format = input_desc->GetOriginFormat();
    OP_LOGD(op_type, "input format is [%s], origin format is [%s].",
            to_string(in_format).c_str(), to_string(ori_format).c_str());
    // only adjust shape for fractal_z, fractal_z_3d
    if (in_format == FORMAT_FRACTAL_Z || in_format == FORMAT_FRACTAL_Z_3D) {
      vector<vector<int64_t>> input_origin_shapes = GetInputOriginShapes(op_paras);
      OP_TILING_CHECK(!AdjustShape(op_type, input_shapes, input_origin_shapes, input_num, in_format, ori_format),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "adjust shape failed"), return false);
    }
  }

  // the ConcatDslTilingHandler need the axis must be > 0, so need do remaining in op
  if (axis < 0) {
    axis = axis + GetDimensionSize(input_shapes);
  }
  OP_LOGD(op_type, "concat axis = %d.", axis);

  vector<vector<int32_t>> axis_list = {{axis}};
  DataType data_type = input_desc->GetDataType();
  OpInfo concat_info(input_shapes, data_type, axis_list);
  // do ConcatDslTilingHandler ->DoTiling
  OP_TILING_CHECK(!op_info.outer_compile_info->DoTiling(op_paras, run_info, concat_info),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "call DoTiling failed"), return false);
  OP_LOGD(op_type, "ConcatDSLTiling end.");
  return true;
}

static string to_string(const ByteBuffer& tiling_data) {
  auto data = tiling_data.str();
  string result;
  const int64_t* data_addr = reinterpret_cast<const int64_t*>(data.c_str());
  for (size_t i = 0; i < data.length() / sizeof(int64_t); i++) {
    result += std::to_string(*data_addr);
    data_addr++;
    result += " ";
  }

  return result;
}

static string to_string(const vector<vector<int64_t>>& shapes) {
  std::string shapes_string = "[";
  for (const auto& shape : shapes) {
    shapes_string += ge::DebugString<int64_t>(shape);
    shapes_string += ",";
  }

  shapes_string += "]";
  return shapes_string;
}

static bool CheckParams(const string& op, const vector<vector<int64_t>>& input_shapes, int32_t dim) {
  if (input_shapes.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op, "The input count should be more than 0");
    return false;
  }

  auto shape_length = input_shapes[0].size();
  auto not_equal_shape_length = [shape_length](const std::vector<int64_t> input_shape) {
      return input_shape.size() != shape_length;
  };
  if (std::any_of(input_shapes.begin(), input_shapes.end(), not_equal_shape_length)) {
      VECTOR_INNER_ERR_REPORT_TILIING(op, "The length of each shape must be equal");
      return false;
  }

  int max_dim = static_cast<int32_t>(shape_length);
  if (dim >= max_dim or dim < -max_dim) {
    VECTOR_INNER_ERR_REPORT_TILIING(op, "the parameter[%s] should be [between %d and %d], but actually is [%d].",
                                    "concat_dim", min(max_dim - 1, -max_dim), max(max_dim - 1, -max_dim), dim);
    return false;
  }

  size_t new_dims = dim >= 0 ? dim : max_dim + dim;
  const auto& shape = input_shapes[0];
  for (size_t i = 1; i < input_shapes.size(); i++) {
    for (size_t j = 0; j < shape_length; j++) {
      if (j != new_dims && input_shapes[i][j] != shape[j]) {
        VECTOR_INNER_ERR_REPORT_TILIING(op, "dims must equal except concat dim[%d], input_values[%s]", dim,
                                        to_string(input_shapes).c_str());
        return false;
      }
    }
  }

  return true;
}

struct TilingParam {
  int64_t axis = 0;
  int64_t out_dims = 1;
  int64_t max_inner_dims = 0;
  int64_t min_inner_dims = 0;
  int64_t output_inner_length = 1;
  int64_t input_count = 0;
  int64_t reserve1 = 0;
  int64_t reserve2 = 0;

  // list of pair with inner_dims and output_idx
  vector<pair<int64_t, int64_t>> input_tiling_info;
  int index = 0;
  void encode(ByteBuffer& tiling_data) {
    ByteBufferPut(tiling_data, axis);
    ByteBufferPut(tiling_data, out_dims);
    ByteBufferPut(tiling_data, max_inner_dims);
    ByteBufferPut(tiling_data, min_inner_dims);
    ByteBufferPut(tiling_data, output_inner_length);
    ByteBufferPut(tiling_data, input_count);
    ByteBufferPut(tiling_data, reserve1);
    ByteBufferPut(tiling_data, reserve2);

    for (const auto& item : input_tiling_info) {
      ByteBufferPut(tiling_data, item.first);
      ByteBufferPut(tiling_data, item.second);
    }
  }
};

static bool GetTilingParam(const vector<vector<int64_t>>& input_shapes,
                           int32_t concat_dim, TilingParam& tiling_param) {
  if (input_shapes.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING("concat", "The input count should be more than 0");
    return false;
  }

  if (concat_dim < 0) {
    concat_dim += input_shapes[0].size();
  }

  auto input_count = input_shapes.size();
  tiling_param.max_inner_dims = 0;
  tiling_param.axis = 1;
  tiling_param.input_count = input_shapes.size();
  tiling_param.out_dims =
      accumulate(input_shapes[0].begin(), input_shapes[0].begin() + concat_dim, 1, std::multiplies<int64_t>());
  tiling_param.min_inner_dims =
      accumulate(input_shapes[0].begin() + concat_dim, input_shapes[0].end(), (int64_t)1, multiplies<int64_t>());
  int64_t output_index = 0;
  for (size_t i = 0; i < input_count; i++) {
    auto inner_dims =
        accumulate(input_shapes[i].begin() + concat_dim, input_shapes[i].end(), (int64_t)1, multiplies<int64_t>());
    tiling_param.max_inner_dims = max(tiling_param.max_inner_dims, inner_dims);
    tiling_param.min_inner_dims = min(tiling_param.min_inner_dims, inner_dims);

    tiling_param.input_tiling_info.emplace_back(pair<int64_t, int64_t>(inner_dims, output_index));

    output_index += inner_dims;
  }

  tiling_param.output_inner_length = output_index;

  return true;
}

static void AdjustParams(const std::string& op_type, vector<vector<int64_t>>& input_shapes, int32_t& concat_dim) {
  if (op_type != "Pack") {
    return;
  }

  if (concat_dim < -1) {
    concat_dim += 1;
    return;
  }

  if (!input_shapes.empty()) {
    int32_t shape_length = static_cast<int32_t>(input_shapes[0].size());
    if (concat_dim == -1 || concat_dim == shape_length) {
      for (auto& shape : input_shapes) {
        shape.push_back(1);
      }
    }
  }
}

bool ConcatTIKTiling(const std::string& op_type, const ge::Operator& op_paras, const ConcatCompileInfo& op_info,
                     utils::OpRunInfo& run_info) {
  OP_LOGD(op_type, "ConcatTIKTiling running.");
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  PROFILING_TILING_INIT(op_type.c_str());

  vector< vector<int64_t> > input_shapes = GetInputShapes(op_paras);
  if (input_shapes.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input shapes failed.");
    return false;
  }
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  int32_t concat_dim = op_info.ori_axis;
  int32_t input_size = op_info.input_size;
  int32_t block_dim = op_info.core_num;

  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  AdjustParams(op_type, input_shapes, concat_dim);
  OP_LOGD(op_type, "to check params.");
  if (!CheckParams(op_type, input_shapes, concat_dim)) {
    return false;
  }

  if (input_size != static_cast<int32_t>(input_shapes.size())) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                    "check input size failed. "
                                    "Input_size in compile is %d, but in params is %zd",
                                    input_size, input_shapes.size());
    return false;
  }

  OP_LOGD(op_type, "GetTilingParam.");
  TilingParam tiling_param;
  if (!GetTilingParam(input_shapes, concat_dim, tiling_param)) {
    return false;
  }

  OP_LOGD(op_type, "encode TilingParam.");
  tiling_param.encode(run_info.GetAllTilingData());
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  OP_LOGD(op_type, "TilingParam:%s.", to_string(run_info.GetAllTilingData()).c_str());

  // block_dim, not need for concat tiling
  run_info.SetBlockDim(block_dim);
  OP_LOGD(op_type, "tiling run success.");
  PROFILING_TILING_END();

  return true;
}

/*
 * @brief: tiling function of op
 * @param [in] op_type: op_type of the op
 * @param [in] op_paras: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool ConcatTiling(const std::string& op_type, const ge::Operator& op_paras, const ConcatCompileInfo& op_info,
                  utils::OpRunInfo& run_info) {
  if (op_info.is_tik) {
    OP_TILING_CHECK(!ConcatTIKTiling(op_type, op_paras, op_info, run_info),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "call TIKTiling failed"), return false);
  } else {
    OP_TILING_CHECK(!ConcatDSLTiling(op_type, op_paras, op_info, run_info),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "call DSLTiling failed"), return false);
  }

  return true;
}

// register tiling interface of the Concat, ConcatV2 op.
REGISTER_OP_TILING_V3_CUSTOM(ConcatD, ConcatTiling, ConcatParseFunc, ConcatCompileInfo);
REGISTER_OP_TILING_V3_CUSTOM(ConcatV2D, ConcatTiling, ConcatParseFunc, ConcatCompileInfo);
// register tiling interface of the Pack op.
REGISTER_OP_TILING_V3_CUSTOM(Pack, ConcatTiling, ConcatParseFunc, ConcatCompileInfo);
}  // namespace optiling
