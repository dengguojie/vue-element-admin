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

#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/storage_shape.h"

using namespace gert;
namespace ops {
ge::graphStatus InferShapeForShape(InferShapeContext *context) {
  auto x_shape = context->GetInputShape(0);
  auto y_shape = context->GetOutputShape(0);
  if ((x_shape == nullptr) || (y_shape == nullptr)) {
    return ge::GRAPH_FAILED;
  }

  y_shape->SetDimNum(1);
  y_shape->SetDim(0, static_cast<int64_t>(x_shape->GetDimNum()));

  return ge::GRAPH_SUCCESS;
}
IMPL_OP(Shape).InferShape(InferShapeForShape);
}  // namespace ops

