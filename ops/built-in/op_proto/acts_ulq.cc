/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

/*!
 * \file acts_ulq.cpp
 * \brief
 */
#include <vector>
#include <string>

#include "math_ops.h"
#include "op_log.h"
#include "axis_util.h"

namespace ge {

IMPLEMT_VERIFIER(ActsULQ, ActsULQVerify) {
    DataType x = op.GetInputDesc("x").GetDataType();
    DataType clamp_min = op.GetInputDesc("clamp_min").GetDataType();
    DataType clamp_max = op.GetInputDesc("clamp_max").GetDataType();
    if (x != clamp_min) {
        return GRAPH_FAILED;
    }

    if (x != clamp_max) {
        return GRAPH_FAILED;
    }


    return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ActsULQInferShape) {
  Shape x_shape = op.GetInputDesc("x").GetShape();
  TensorDesc y = op.GetOutputDesc("y");
  y.SetShape(x_shape);
  y.SetDataType(op.GetInputDesc("x").GetDataType());
  CHECK(op.UpdateOutputDesc("y", y) != GRAPH_SUCCESS,
        GE_OP_LOGE(GRAPH_FAILED, "Update output desc of node[ActsULQInferShape] failed."), return GRAPH_FAILED);

  TensorDesc clamp_min_mask = op.GetOutputDesc("clamp_min_mask");
  clamp_min_mask.SetShape(x_shape);
  CHECK(op.UpdateOutputDesc("clamp_min_mask", clamp_min_mask) != GRAPH_SUCCESS,
        GE_OP_LOGE(GRAPH_FAILED, "Update output desc of node[ActsULQInferShape] failed."), return GRAPH_FAILED);

  TensorDesc clamp_max_mask = op.GetOutputDesc("clamp_max_mask");
  clamp_max_mask.SetShape(x_shape);
  CHECK(op.UpdateOutputDesc("clamp_max_mask", clamp_max_mask) != GRAPH_SUCCESS,
        GE_OP_LOGE(GRAPH_FAILED, "Update output desc of node[ActsULQInferShape] failed."), return GRAPH_FAILED);

  TensorDesc x_clamped_loss = op.GetOutputDesc("x_clamped_loss");
  x_clamped_loss.SetShape(x_shape);
  CHECK(op.UpdateOutputDesc("x_clamped_loss", x_clamped_loss) != GRAPH_SUCCESS,
        GE_OP_LOGE(GRAPH_FAILED, "Update output desc of node[ActsULQInferShape] failed."), return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(ActsULQ, ActsULQInferShape);

// Registered verify function
VERIFY_FUNC_REG(ActsULQ, ActsULQVerify);
}  // namespace ge

