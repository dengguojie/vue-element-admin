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
ge::graphStatus InferShapeForUnsqueeze(gert::InferShapeContext* context) {
  const auto x_shape = context->GetInputShape(0);
  auto y_shape = context->GetOutputShape(0);
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  GE_CHECK_NOTNULL(attrs);
  auto axes = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(0);

  if (x_shape == nullptr || axes == nullptr || y_shape == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "X_shape or axes or y_shape pointer must not be nullptr!");
    return ge::GRAPH_FAILED;
  }

  const auto dim_size = x_shape->GetDimNum();
  const auto out_dim_size = dim_size + axes->GetSize();
  y_shape->SetDimNum(out_dim_size);
  if (out_dim_size > y_shape->kMaxDimNum) {
    GELOGE(ge::GRAPH_FAILED, "DimNum of output shape is %zu, larger than kMaxDimNum which is %zu!", out_dim_size,
           (y_shape->kMaxDimNum));
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0UL; i < out_dim_size; ++i) {
    y_shape->SetDim(i, 0);
  }

  for (size_t i = 0UL; i < axes->GetSize(); ++i) {
    const int64_t axis = (axes->GetData())[i];
    const int64_t real_axis = (axis < 0) ? (axis + static_cast<int64_t>(out_dim_size)) : axis;
    if ((real_axis < 0) || (real_axis >= static_cast<int64_t>(out_dim_size))) {
      GELOGE(ge::GRAPH_FAILED, "Unsqueeze axis must be in range [-%zu, %zu)!", out_dim_size, out_dim_size);
      return ge::GRAPH_FAILED;
    }
    if (y_shape->GetDim(real_axis) == 1) {
      GELOGE(ge::GRAPH_FAILED, "Unsqueeze axis repeated!");
      return ge::GRAPH_FAILED;
    }
    y_shape->SetDim(real_axis, 1);
  }

  size_t idx = 0UL;
  for (size_t i = 0UL; i < out_dim_size; ++i) {
    if (y_shape->GetDim(i) != 1) {
      y_shape->SetDim(i, x_shape->GetDim(idx));
      ++idx;
    }
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(Unsqueeze).InferShape(InferShapeForUnsqueeze);
}  // namespace ops