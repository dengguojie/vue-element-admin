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
#include "runtime_util.h"

using namespace ge;
namespace ops {
static void GetRangeConstValue(const gert::Tensor *const_tensor, const DataType &dtype,
                               std::vector<float> &const_data) {
  uint32_t size = 0;
  if (dtype == ge::DT_INT32) {
    const int32_t *const_data_ptr = const_tensor->GetData<int32_t>();
    size = const_tensor->GetShapeSize();
    for (size_t i = 0; i < size; i++) {
      const_data.push_back((int32_t)(*(const_data_ptr + i)));
    }
  } else if (dtype == ge::DT_FLOAT) {
    const float *const_data_ptr = const_tensor->GetData<float>();
    size = const_tensor->GetShapeSize();
    for (size_t i = 0; i < size; i++) {
      const_data.push_back((float)(*(const_data_ptr + i)));
    }
  } else if (dtype == ge::DT_INT64) {
    const int64_t *const_data_ptr = const_tensor->GetData<int64_t>();
    size = const_tensor->GetShapeSize();
    for (size_t i = 0; i < size; i++) {
      const_data.push_back((int64_t)(*(const_data_ptr + i)));
    }
  } else if (dtype == ge::DT_DOUBLE) {
    const double *const_data_ptr = const_tensor->GetData<double>();
    size = const_tensor->GetShapeSize();
    for (size_t i = 0; i < size; i++) {
      const_data.push_back((double)(*(const_data_ptr + i)));
    }
  }
  return;
}

int CalculateDimValue(DataType &start_dtype, DataType &limit_dtype,
                      DataType &delta_dtype, DataType type) {
  if ((start_dtype == type) && (limit_dtype == type) && (delta_dtype == type)) {
    return true;
  }

  return false;
}

ge::graphStatus RangeInferShapeFunc(gert::InferShapeContext *context) {
  auto start_tensor = context->GetInputTensor(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, start_tensor);
  auto limit_tensor = context->GetInputTensor(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, limit_tensor);
  auto delta_tensor = context->GetInputTensor(2);
  OPS_CHECK_NULL_WITH_CONTEXT(context, delta_tensor);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  std::vector<float> start_multiples;
  std::vector<float> limit_multiples;
  std::vector<float> delta_multiples;
  DataType start_dtype = start_tensor->GetDataType();
  DataType limit_dtype = limit_tensor->GetDataType();
  DataType delta_dtype = delta_tensor->GetDataType();

  GetRangeConstValue(start_tensor, start_dtype, start_multiples);
  GetRangeConstValue(limit_tensor, limit_dtype, limit_multiples);
  GetRangeConstValue(delta_tensor, delta_dtype, delta_multiples);
  if (start_multiples.empty() || limit_multiples.empty() || delta_multiples.empty()) {
    out_shape->SetDimNum(1);
    out_shape->SetDim(0, UNKNOWN_DIM);
  }

  float assist_num = std::abs(limit_multiples[0] - start_multiples[0]);
  float assist_num_one = std::abs(delta_multiples[0]);
  int res = 0;
  if ((assist_num_one < 1e-6) || (assist_num_one == 0)) {
    return GRAPH_FAILED;
  }

  if (CalculateDimValue(start_dtype, limit_dtype, delta_dtype, ge::DT_INT32)) {
    res = static_cast<int>(ceil(float(assist_num) / assist_num_one));
  } else if (CalculateDimValue(start_dtype, limit_dtype, delta_dtype, ge::DT_INT64)) {
    res = static_cast<int>(ceil(float(assist_num) / assist_num_one));
  } else if (CalculateDimValue(start_dtype, limit_dtype, delta_dtype, ge::DT_DOUBLE)) {
    res = static_cast<int>(ceil(double(assist_num) / assist_num_one));
  } else {
    res = static_cast<int>(ceil(assist_num / assist_num_one));
  }

  out_shape->SetDimNum(1);
  out_shape->SetDim(0, res);

  return GRAPH_SUCCESS;
}

IMPL_OP(Range)
    .InputsDataDependency({0, 1, 2})
    .InferShape(RangeInferShapeFunc);
}  // namespace ops
