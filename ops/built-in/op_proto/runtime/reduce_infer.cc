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
template<typename T>
ge::graphStatus ReduceDims(const T *axes_dims, int32_t axes_size, const gert::Shape *x_shape,
                           gert::Shape *output_shape, const bool keep_dims) {
  T dim_num = x_shape->GetDimNum();
  if (keep_dims) {
    *output_shape = *x_shape;
    for (int32_t i = 0; i < axes_size; i++) {
      T dim = axes_dims[i] < 0 ? axes_dims[i] + dim_num : axes_dims[i];
      if (dim < 0 || dim >= dim_num) return ge::GRAPH_FAILED;
      output_shape->SetDim(dim, 1);
    }
  } else {
    bool reduce = false;
    for (T j = 0; j < dim_num; j++, reduce = false) {
      for (int32_t i = 0; i < axes_size; i++) {
        T dim = axes_dims[i] < 0 ? axes_dims[i] + dim_num : axes_dims[i];
        if (dim < 0 || dim >= dim_num) return ge::GRAPH_FAILED;
        if (dim == j) {
          reduce = true;
          break;
        }
      }
      if (!reduce) output_shape->AppendDim(x_shape->GetDim(j));
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForReduceCommon(gert::InferShapeContext *context) {
  auto in_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  auto axes_tensor = context->GetInputTensor(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, axes_tensor);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

  const bool *keep_dims = attrs->GetAttrPointer<bool>(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, keep_dims);

  auto axes_size = static_cast<int32_t>(axes_tensor->GetShapeSize());
  OP_CHECK(axes_size < 1,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "axes size must big than 0!"),
           return ge::GRAPH_FAILED);

  if (axes_tensor->GetDataType() == ge::DT_INT32) {
    auto axes_dims = axes_tensor->GetData<int32_t>();
    return ReduceDims<int32_t>(axes_dims, axes_size, in_shape, out_shape, *keep_dims);
  }
  auto axes_dims = axes_tensor->GetData<int64_t>();
  return ReduceDims<int64_t>(axes_dims, axes_size, in_shape, out_shape, *keep_dims);
}

IMPL_OP(ReduceSum)
    .InferShape(InferShapeForReduceCommon)
    .InputsDataDependency({1});
}  // namespace ops
