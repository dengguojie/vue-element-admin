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
int64_t GetPartSize(const gert::Shape *shape, int64_t begin, int64_t end) {
  int64_t size = 1;
  for (int64_t i = begin; i < end; i++) {
    size *= shape->GetDim(i);
  }
  return size;
}

ge::graphStatus InferShapeForFlatten(gert::InferShapeContext *context) {
  auto in_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

  const int64_t *p_axis = attrs->GetAttrPointer<int64_t>(0);
  const int64_t x_dim = in_shape->GetDimNum();
  if (*p_axis < -x_dim || *p_axis > x_dim) {
    string err_msg1 = ConcatString("axis ", *p_axis, " out of range: [-", x_dim, ", ", x_dim, "]");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), err_msg);
    return GRAPH_FAILED;
  }
  int64_t axis = (*p_axis >= 0) ? *p_axis : (x_dim + *p_axis);

  out_shape->AppendDim(GetPartSize(in_shape, 0, axis));
  out_shape->AppendDim(GetPartSize(in_shape, axis, x_dim));
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(Flatten)
    .InferShape(InferShapeForFlatten);
}  // namespace ops
