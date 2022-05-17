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
bool InferBroadcastshapeForStatic(const gert::Shape *shape_x, const gert::Shape *shape_y, gert::Shape *shape_output) {
  auto shape_x_len = shape_x->GetDimNum();
  auto shape_y_len = shape_y->GetDimNum();
  if (shape_x_len >= shape_y_len) {
    // when inputx len >= inputy len
    // input_x = [128, 128, 128] Vs input_y = [128]
    auto len_sub = shape_x_len - shape_y_len;
    *shape_output = *shape_x;
    for (size_t i = 0; i < shape_y_len; i++) {
      int64_t dim_size = std::max(shape_x->GetDim(len_sub + i), shape_y->GetDim(i));
      // if one dim is 0, the output dim is 0
      dim_size = (shape_x->GetDim(len_sub + i) == 0 || shape_y->GetDim(i) == 0) ? 0 : dim_size;
      shape_output->SetDim(len_sub + i, dim_size);
    }
  } else {
    // when inputx len < inputy len
    // input_x = [128] Vs input_y = [128, 128, 128]
    auto len_sub = shape_y_len - shape_x_len;
    *shape_output = *shape_y;
    for (size_t i = 0; i < shape_x_len; i++) {
      int64_t dim_size = std::max(shape_y->GetDim(len_sub + i), shape_x->GetDim(i));
      // if one dim is 0, the output dim is 0
      dim_size = (shape_y->GetDim(len_sub + i) == 0 || shape_x->GetDim(i) == 0) ? 0 : dim_size;
      shape_output->SetDim(len_sub + i, dim_size);
    }
  }
  return true;
}
ge::graphStatus InferShapeForTwoInOneOut(gert::InferShapeContext *context) {
  auto in_shape1 = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape1);
  auto in_shape2 = context->GetInputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape2);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  OP_CHECK(!InferBroadcastshapeForStatic(in_shape1, in_shape2, out_shape),
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "broadcast shape error!"),
           return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP(Add)
    .InferShape(InferShapeForTwoInOneOut);
IMPL_OP(Mul)
    .InferShape(InferShapeForTwoInOneOut);
}  // namespace ops
