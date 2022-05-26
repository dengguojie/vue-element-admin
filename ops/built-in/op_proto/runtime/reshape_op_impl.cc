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

namespace ops {
template<typename T>
ge::graphStatus ReshapeInferShapeImpl(const T *reshape_dims, const gert::Shape &x_shape, gert::Shape &output_shape,
                                      int32_t reshape_size) {
  constexpr T kUnknownDim = -1;
  output_shape.SetDimNum(reshape_size);
  auto x_shape_size = x_shape.GetShapeSize();
  int64_t output_shapesize = 1;
  size_t unknown_dim_idx = std::numeric_limits<size_t>::max();
  for (int32_t i = 0; i < reshape_size; i++) {
    if (reshape_dims[i] != kUnknownDim) {
      output_shape.SetDim(i, reshape_dims[i]);
      output_shapesize *= reshape_dims[i];
    } else {
      output_shape.SetDim(i, 1);
      unknown_dim_idx = i;
    }
  }
  if (unknown_dim_idx == std::numeric_limits<size_t>::max() && output_shapesize == x_shape_size) {
    return ge::GRAPH_SUCCESS;
  } else if (unknown_dim_idx != std::numeric_limits<size_t>::max() && x_shape_size % output_shapesize == 0) {
    output_shape.SetDim(unknown_dim_idx, x_shape_size / output_shapesize);
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

ge::graphStatus InferShapeForReshape(gert::InferShapeContext *context) {
  auto x_shape = context->GetInputShape(0);
  auto shape_tensor = context->GetInputTensor(1);
  auto output_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);
  OPS_CHECK_NULL_WITH_CONTEXT(context, shape_tensor);
  OPS_CHECK_NULL_WITH_CONTEXT(context, output_shape);

  auto reshape_size = static_cast<int32_t>(shape_tensor->GetShapeSize());
  if (reshape_size < 1) {
    return ge::GRAPH_FAILED;
  }
  if (shape_tensor->GetDataType() == ge::DT_INT32) {
    auto reshape_data = shape_tensor->GetData<int32_t>();
    return ReshapeInferShapeImpl<int32_t>(reshape_data, *x_shape, *output_shape, reshape_size);
  } else {
    auto reshape_data = shape_tensor->GetData<int64_t>();
    return ReshapeInferShapeImpl<int64_t>(reshape_data, *x_shape, *output_shape, reshape_size);
  }
}
IMPL_OP(Reshape).InferShape(InferShapeForReshape).InputsDataDependency({1});
}  // namespace ops