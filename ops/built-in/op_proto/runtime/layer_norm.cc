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
#include "op_util.h"

using namespace ge;
namespace ops {
constexpr size_t LAYERNORM_IDX_IN_X = 0;
constexpr size_t LAYERNORM_IDX_OUT_Y = 0;
constexpr size_t LAYERNORM_IDX_OUT_MEAN = 1;
constexpr size_t LAYERNORM_IDX_OUT_VAR = 2;
constexpr size_t LAYERNORM_ATTR_IDX_BEGIN_NORM_AXIS = 0;

ge::graphStatus LayerNormInferShape(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do LayerNormInferShape");

  // get input shapes
  const gert::Shape* x_shape = context->GetInputShape(LAYERNORM_IDX_IN_X);
  OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);

  // get output shapes
  gert::Shape* y_shape = context->GetOutputShape(LAYERNORM_IDX_OUT_Y);
  OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);
  gert::Shape* mean_shape = context->GetOutputShape(LAYERNORM_IDX_OUT_MEAN);
  OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);
  gert::Shape* var_shape = context->GetOutputShape(LAYERNORM_IDX_OUT_VAR);
  OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);

  size_t real_dim_num = x_shape->GetDimNum();
  *y_shape = *x_shape;
  mean_shape->SetDimNum(real_dim_num);
  var_shape->SetDimNum(real_dim_num);

  // process attr
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const int64_t* begin_norm_axis = attrs->GetAttrPointer<int64_t>(LAYERNORM_ATTR_IDX_BEGIN_NORM_AXIS);
  OPS_CHECK_NULL_WITH_CONTEXT(context, begin_norm_axis);

  OP_CHECK(!IsDimValid(real_dim_num, *begin_norm_axis),
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                               GenInvalidDimMsg("begin_norm_axis", real_dim_num, *begin_norm_axis)),
           return ge::GRAPH_FAILED);

  int64_t new_begin_norm_axis =
      *begin_norm_axis < 0 ? *begin_norm_axis + static_cast<int64_t>(real_dim_num) : *begin_norm_axis;

  // update mean and var shape
  for (size_t i = 0; i < real_dim_num; ++i) {
    if (static_cast<int64_t>(i) >= new_begin_norm_axis) {
      mean_shape->SetDim(i, 1);
      var_shape->SetDim(i, 1);
    } else {
      mean_shape->SetDim(i, x_shape->GetDim(i));
      var_shape->SetDim(i, x_shape->GetDim(i));
    }
  }

  OP_LOGD(context->GetNodeName(), "End to do LayerNormInferShape");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(LayerNorm).InferShape(LayerNormInferShape);
}  // namespace ops
