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
#include "layer_norm_x_backprop_v2.h"
#include "runtime2_util.h"
#include "op_tiling_util.h"
#include "op_util.h"

using namespace ge;

namespace optiling {
static bool LayerNormXBackpropV2ParseFunc(const std::string& node_name, const nlohmann::json& compile_info,
                                          LayerNormXBackpropV2CompileInfo* compile_value) {
  compile_value->dsl_compile_info = ParseAutoTiling("LayerNormXBackpropV2", compile_info);
  OP_TILING_CHECK(compile_value->dsl_compile_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(node_name, "CreateAutoTilingHandler return nullptr"), return false);
  if (compile_info.count("reduce_mean_cof") > 0) {
    OP_TILING_CHECK(!ReadCompileItem(compile_info, "reduce_mean_cof", compile_value->reduce_mean_cof),
                    VECTOR_INNER_ERR_REPORT_TILIING(node_name, "LayerNormParseFunc, get reduce_mean_cof error"),
                    return false);
  }
  // add for unknown axis mode
  (void)ReadCompileItem(compile_info, "unknown_mode", compile_value->unknown_mode, false);
  return true;
}

ge::graphStatus TilingPrepareForLayerNormXBackpropV2(gert::TilingParseContext* context) {
  auto compile_info = GetCompileInfoPtr<LayerNormXBackpropV2CompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetCompileInfoJson(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);

  OP_TILING_CHECK(!LayerNormXBackpropV2ParseFunc(context->GetNodeName(), *parsed_object_cinfo, compile_info),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                  "do TilingPrepareForLayerNormXBackpropV2 failed"),
                  return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

bool GetReduceAxis(const gert::Shape& input_x_shape, const gert::Shape& input_mean_shape,
                   ge::Format input_x_format, std::vector<int64_t>& reduce_axis,
                   std::vector<int64_t>& shape_x_nz) {
  const std::size_t rank = input_x_shape.GetDimNum();
  if (input_x_format == FORMAT_FRACTAL_NZ) {
    size_t nz_begin = rank - 4;
    for (size_t i = 0; i < nz_begin; i++) {
      shape_x_nz.push_back(input_x_shape.GetDim(i));
    }
    shape_x_nz.push_back(input_x_shape.GetDim(nz_begin));
    shape_x_nz.push_back(input_x_shape.GetDim(nz_begin+1));
    shape_x_nz.push_back(input_x_shape.GetDim(nz_begin+2));
    shape_x_nz.push_back(input_x_shape.GetDim(nz_begin+2));

    std::vector<int64_t> shape_mean_nz = {};
    const std::size_t len_mean = input_mean_shape.GetDimNum();
    size_t mean_nz_begin = len_mean - 2;
    for (size_t i = 0; i < mean_nz_begin; i++) {
      shape_mean_nz.push_back(input_mean_shape.GetDim(i));
    }
    shape_mean_nz.push_back(1);
    shape_mean_nz.push_back(input_x_shape.GetDim(nz_begin+1));
    shape_mean_nz.push_back(input_x_shape.GetDim(nz_begin+2));
    shape_mean_nz.push_back(1);

    size_t x_nz_size = shape_x_nz.size();
    for (size_t i = 0; i < x_nz_size; i++) {
      if (shape_x_nz[i] != shape_mean_nz[i]) {
        reduce_axis.push_back(static_cast<int64_t>(i));
      }
    }
  } else {
    for (size_t i = 0; i < rank; i++) {
      int64_t xtem = input_x_shape.GetDim(i);
      int64_t mean = input_mean_shape.GetDim(i);
      if (xtem != mean) {
        reduce_axis.push_back(static_cast<int64_t>(i));
      }
    } 
  }
  return true;
}

static ge::graphStatus LayerNormXBackpropV2UnknownAxisTiling(gert::TilingContext* context,
                                                             const LayerNormXBackpropV2CompileInfo* compile_info) {
  OP_LOGD(context->GetNodeName(), "LayerNormXBackpropV2UnknownAxisTiling running.");
  const gert::StorageShape* input_x_shape_cls = context->GetInputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input_x_shape_cls);
  const gert::StorageShape* input_mean_shape_cls = context->GetInputShape(3);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input_mean_shape_cls);

  const gert::Tensor* input_x_tenser = context->GetInputTensor(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input_x_tenser);

  const gert::Shape& input_x_shape = input_x_shape_cls->GetStorageShape();
  const gert::Shape& input_mean_shape = input_mean_shape_cls->GetStorageShape();
  ge::Format x_format = context->GetInputDesc(1)->GetStorageFormat();

  std::vector<int64_t> reduce_axis = {};
  std::vector<int64_t> shape_x_nz = {};
  GetReduceAxis(input_x_shape, input_mean_shape, x_format, reduce_axis, shape_x_nz);

  const std::vector<gert::Shape> input_gert_shapes = {input_x_shape};
  ge::DataType input_x_dtype = input_x_tenser->GetDataType();
  OpInfo norm_info(compile_info->dsl_compile_info.get());
  norm_info.SetInputShape(&input_gert_shapes);
  norm_info.SetInputType(&input_x_dtype);
  norm_info.SetAxes(&reduce_axis);
  OP_TILING_CHECK(!DoAutoTiling(context, &norm_info),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "call DoTiling failed"),
                  return ge::GRAPH_FAILED);
  
  if (compile_info->reduce_mean_cof) {
    float mean_num = 1.0;
    int32_t reduce_axis_size = reduce_axis.size();
    if (x_format == FORMAT_FRACTAL_NZ) {
      for (int32_t i = 0; i < reduce_axis_size; i++) {
        mean_num *= shape_x_nz[reduce_axis[i]];
      }
    } else {
      for (int32_t i = 0; i < reduce_axis_size; i++) {
        mean_num *= input_x_shape.GetDim(reduce_axis[i]);
      }
    }
    
    float mean_cof = pow(mean_num, -1);
    float mean_cof2 = pow(mean_num, -1) * 2;
    
    auto tiling_data = context->GetTilingData<LayerNormXBackpropV2TilingData>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, tiling_data);

    tiling_data->mean_cof = mean_cof;
    tiling_data->mean_cof2 = mean_cof2;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForLayerNormXBackpropV2(gert::TilingContext* context) {
  const LayerNormXBackpropV2CompileInfo* compile_info =
      reinterpret_cast<const LayerNormXBackpropV2CompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  if (compile_info->unknown_mode) {
    return LayerNormXBackpropV2UnknownAxisTiling(context, compile_info); 
  }
  OpInfo norm_info(compile_info->dsl_compile_info.get());
  OP_TILING_CHECK(!DoAutoTiling(context, &norm_info),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "call DoTiling failed"),
                  return ge::GRAPH_FAILED);
  
  if (compile_info->reduce_mean_cof) {
    const gert::StorageShape* input_x_shape_cls = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, input_x_shape_cls);
    const gert::StorageShape* input_mean_shape_cls = context->GetInputShape(3);
    OPS_CHECK_NULL_WITH_CONTEXT(context, input_mean_shape_cls);

    const gert::Shape& input_x_shape = input_x_shape_cls->GetStorageShape();
    const gert::Shape& input_mean_shape = input_mean_shape_cls->GetStorageShape();
    ge::Format x_format = context->GetInputDesc(1)->GetStorageFormat();

    std::vector<int64_t> reduce_axis = {};
    std::vector<int64_t> shape_x_nz = {};
    GetReduceAxis(input_x_shape, input_mean_shape, x_format, reduce_axis, shape_x_nz);
    
    float mean_num = 1.0;
    int32_t reduce_axis_size = reduce_axis.size();
    if (x_format == FORMAT_FRACTAL_NZ) {
      for (int32_t i = 0; i < reduce_axis_size; i++) {
        mean_num *= shape_x_nz[reduce_axis[i]];
      }
    } else {
      for (int32_t i = 0; i < reduce_axis_size; i++) {
        mean_num *= input_x_shape.GetDim(reduce_axis[i]);
      }
    }
    
    float mean_cof = pow(mean_num, -1);
    float mean_cof2 = pow(mean_num, -1) * 2;
    auto tiling_data = context->GetTilingData<LayerNormXBackpropV2TilingData>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, tiling_data);

    tiling_data->mean_cof = mean_cof;
    tiling_data->mean_cof2 = mean_cof2;
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(LayerNormXBackpropV2)
    .Tiling(TilingForLayerNormXBackpropV2)
    .TilingParse<LayerNormXBackpropV2CompileInfo>(TilingPrepareForLayerNormXBackpropV2);
}  // namespace optiling
