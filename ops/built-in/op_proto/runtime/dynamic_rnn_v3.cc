/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
constexpr int PROJECT_SHAPE_SIZE = 2;
constexpr int X_SHAPE_SIZE = 3;

ge::graphStatus InferShapeForDynamicRNNV3(gert::InferShapeContext *context) {
  auto x_shape = context->GetInputShape(0);
  auto w_shape = context->GetInputShape(1);
  if (x_shape == nullptr || w_shape == nullptr) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT("DynamicRNNV3", "get input info error!");
    return ge::GRAPH_FAILED;
  }

  int64_t stateSize = 0;
  auto project_shape = context->GetOptionalInputShape(11);
  if (project_shape != nullptr) {
    if (project_shape->GetDimNum() < PROJECT_SHAPE_SIZE) {
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT("DynamicRNNV3", "project shape is illegal!");
      return ge::GRAPH_FAILED;
    }
    stateSize = project_shape->GetDim(1);
  }
  if (x_shape->GetDimNum() != X_SHAPE_SIZE) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT("DynamicRNNV3", "The input shape of X not equal 3, please check!");
    return ge::GRAPH_FAILED;
  }

  auto hidden_size = w_shape->GetDim(1) / 4;
  auto num_step = x_shape->GetDim(0);
  auto batch_size = x_shape->GetDim(1);

  auto outputY_shape = context->GetOutputShape(0);
  auto outputH_shape = context->GetOutputShape(1);
  auto outputC_shape = context->GetOutputShape(2);
  auto outputI_shape = context->GetOutputShape(3);
  auto outputJ_shape = context->GetOutputShape(4);
  auto outputF_shape = context->GetOutputShape(5);
  auto outputO_shape = context->GetOutputShape(6);
  auto outputTanhc_shape = context->GetOutputShape(7);

  if (outputY_shape == nullptr || outputH_shape == nullptr || outputC_shape == nullptr || outputI_shape == nullptr ||
      outputJ_shape == nullptr || outputF_shape == nullptr || outputO_shape == nullptr ||
      outputTanhc_shape == nullptr) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT("DynamicRNNV3", "get output info error!");
    return ge::GRAPH_FAILED;
  }

  *outputY_shape = {num_step, batch_size, stateSize};
  *outputH_shape = {num_step, batch_size, stateSize};
  *outputC_shape = {num_step, batch_size, hidden_size};
  *outputI_shape = {num_step, batch_size, hidden_size};
  *outputJ_shape = {num_step, batch_size, hidden_size};
  *outputF_shape = {num_step, batch_size, hidden_size};
  *outputO_shape = {num_step, batch_size, hidden_size};
  *outputTanhc_shape = {num_step, batch_size, hidden_size};

  return ge::GRAPH_SUCCESS;
}

IMPL_OP(DynamicRNNV3)
    .InferShape(InferShapeForDynamicRNNV3);
}  // namespace gert