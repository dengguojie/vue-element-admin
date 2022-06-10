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
ge::graphStatus InferShapeForSqueeze(gert::InferShapeContext* context) {
  const auto x_shape = context->GetInputShape(0);
  auto y_shape = context->GetOutputShape(0);
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  GE_CHECK_NOTNULL(attrs);
  auto axis = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(0);
  if (x_shape == nullptr || axis == nullptr || y_shape == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "X_shape or axis or y_shape pointer must not be nullptr!");
    return ge::GRAPH_FAILED;
  }

  y_shape->SetDimNum(0);
  auto dim_size = x_shape->GetDimNum();
  if (axis->GetSize() == 0UL) {
    // squeeze all 1
    for (size_t index = 0UL; index < dim_size; ++index) {
      if (x_shape->GetDim(index) != 1) {
        y_shape->AppendDim(x_shape->GetDim(index));
      }
    }
    return ge::GRAPH_SUCCESS;
  }
  // squeeze by axes
  bool dim_index[x_shape->kMaxDimNum] = {false};
  for (size_t i = 0UL; i < axis->GetSize(); ++i) {
    const int64_t tmp_axis = (axis->GetData())[i];
    const int64_t real_axis = (tmp_axis < 0) ? (tmp_axis + static_cast<int64_t>(dim_size)) : tmp_axis;
    if ((real_axis < 0) || (real_axis >= static_cast<int64_t>(dim_size))) {
      GELOGE(ge::GRAPH_FAILED, "Squeeze axis must be in range [-%ld, %ld)!", static_cast<int64_t>(dim_size),
             static_cast<int64_t>(dim_size));
      return ge::GRAPH_FAILED;
    }
    if (x_shape->GetDim(real_axis) != 1) {
      GELOGE(ge::GRAPH_FAILED, "Cannot squeeze the shape whose dim was not 1!");
      return ge::GRAPH_FAILED;
    }

    dim_index[real_axis] = true;
  }

  for (size_t i = 0UL; i < dim_size; ++i) {
    if (dim_index[i] != true) {
      y_shape->AppendDim(x_shape->GetDim(i));
    }
  }
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(Squeeze).InferShape(InferShapeForSqueeze);
}  // namespace ops
