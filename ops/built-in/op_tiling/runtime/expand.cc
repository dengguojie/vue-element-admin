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
 * \file expand.cc
 * \brief
 */
#include "expand.h"

#include "error_log.h"
#include "op_tiling_util.h"
#include "runtime2_util.h"

namespace{
const std::string OP_NAME = "Expand";
const size_t INPUT_INDEX_X = 0;
const size_t INPUT_INDEX_SHAPE = 1;
}  // namespace

namespace optiling {
template <typename T>
static void GetConstIntData(const gert::Tensor* tensor, int64_t data_size, std::vector<int64_t>& param) {
  const T* data = tensor->GetData<T>();
  for (int64_t i = 0; i < data_size; i++) {
    param[i] = *(data + i);
  }
  return;
}

static std::vector<int64_t> GetConstShapeValue(const gert::Tensor* tensor) {
  const int64_t shape_size = tensor->GetShapeSize();
  std::vector<int64_t> new_shape(shape_size);
  if (tensor->GetDataType() == ge::DT_INT32) {
    GetConstIntData<int32_t>(tensor, shape_size, new_shape);
  } else if (tensor->GetDataType() == ge::DT_INT64) {
    GetConstIntData<int64_t>(tensor, shape_size, new_shape);
  } else {
    GetConstIntData<int16_t>(tensor, shape_size, new_shape);
  }
  return new_shape;
}

static std::vector<int64_t> GetXShape(const gert::Shape& shape) {
  int32_t dim_num = static_cast<int32_t>(shape.GetDimNum());
  std::vector<int64_t> param(dim_num);
  for (int32_t i = 0; i < dim_num; i++) {
    param[i] = shape.GetDim(i);
  }
  return param;
}

ge::graphStatus ExpandTiling(gert::TilingContext* context) {
  OP_LOGD(OP_NAME.c_str(), "ExpandTiling running begin");
  const gert::Tensor* shape_tensor = context->GetInputTensor(INPUT_INDEX_SHAPE);
  OPS_CHECK_NULL_WITH_CONTEXT(context, shape_tensor);
  std::vector<int64_t> shape_value = GetConstShapeValue(shape_tensor);

  const gert::Tensor* x_tensor = context->GetInputTensor(INPUT_INDEX_X);
  OPS_CHECK_NULL_WITH_CONTEXT(context, x_tensor);
  ge::DataType data_type = x_tensor->GetDataType();
  const gert::Shape x_shape = x_tensor->GetStorageShape();
  std::vector<int64_t> x_runtime_shape = GetXShape(x_shape);
  std::vector<std::vector<int64_t>> shapes = {shape_value, x_runtime_shape};

  auto compile_info = reinterpret_cast<const ExpandCompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info->dsl_compile_info);
  OpInfo expand_info(compile_info->dsl_compile_info.get());
  expand_info.SetInputShape(&shapes);
  expand_info.SetInputType(&data_type);
  OP_TILING_CHECK(!DoAutoTiling(context, &expand_info),
                  VECTOR_INNER_ERR_REPORT_TILIING(OP_NAME.c_str(), "call DoTiling failed"), return ge::GRAPH_FAILED);
  OP_LOGD(OP_NAME.c_str(), "ExpandTiling running end");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForExpand(gert::TilingParseContext* context) {
  OP_LOGD(OP_NAME.c_str(), "TilingPrepareForExpand running.");
  auto compile_info = MutableCompileInfo<ExpandCompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetJsonObj(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);
  compile_info->dsl_compile_info = ParseAutoTiling("Expand", *parsed_object_cinfo);
  OP_TILING_CHECK(compile_info->dsl_compile_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(OP_NAME.c_str(), "CreateExpandTilingHandler failed"),
                  return ge::GRAPH_FAILED);
  OP_LOGD(OP_NAME.c_str(), "TilingPrepareForExpand GRAPH_SUCCESS.");
  return ge::GRAPH_SUCCESS;
}

// register tiling interface of the Expand.
IMPL_OP(Expand).Tiling(ExpandTiling).TilingParse<ExpandCompileInfo>(TilingPrepareForExpand);
}  // namespace optiling
