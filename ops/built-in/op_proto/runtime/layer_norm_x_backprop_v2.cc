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
ge::graphStatus InferShapeForLayerNormXBackpropV2(gert::InferShapeContext *context) {
  OP_LOGD(context->GetNodeName(), "Begin to do LayerNormXBackpropV2InferShape");
  const gert::Shape *input_dy_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input_dy_shape);
  gert::Shape *output_pd_x_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, output_pd_x_shape);
  gert::Shape *output_res_for_gamma_shape = context->GetOutputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, output_res_for_gamma_shape);

  *output_pd_x_shape = *input_dy_shape;
  *output_res_for_gamma_shape = *input_dy_shape;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(LayerNormXBackpropV2)
    .InferShape(InferShapeForLayerNormXBackpropV2);
}  // namespace ops
