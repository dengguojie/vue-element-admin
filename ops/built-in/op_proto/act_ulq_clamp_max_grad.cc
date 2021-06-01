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
 * \file act_ulq_clamp_max_grad.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <string>

#include "math_ops.h"
#include "op_log.h"


namespace ge {

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ActULQClampMaxGradInferShape) {
    Shape y_grad_shape = op.GetInputDesc("y_grad").GetShape();
    Shape clamp_max_mask_shape = op.GetInputDesc("clamp_max_mask").GetShape();
    Shape x_clamped_loss_shape = op.GetInputDesc("x_clamped_loss").GetShape();

    if (y_grad_shape.GetDims() != clamp_max_mask_shape.GetDims()) {
        OP_LOGE(op.GetName().c_str(), "The shape of clamp_max_mask must be the same as y_grad!");
        return GRAPH_FAILED;
    }

    if (y_grad_shape.GetDims() != x_clamped_loss_shape.GetDims()) {
        OP_LOGE(op.GetName().c_str(), "The shape of x_clamped_loss must be the same as y_grad!");
        return GRAPH_FAILED;
    }

    TensorDesc clamp_max_grad = op.GetOutputDesc("clamp_max_grad");
    clamp_max_grad.SetShape({});
    clamp_max_grad.SetDataType(op.GetInputDesc("y_grad").GetDataType());
    if (op.UpdateOutputDesc("clamp_max_grad", clamp_max_grad) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Update output[clamp_max_grad] failed.");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ActULQClampMaxGrad, ActULQClampMaxGradVerify) {
    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(ActULQClampMaxGrad, ActULQClampMaxGradInferShape);

// Registered verify function
VERIFY_FUNC_REG(ActULQClampMaxGrad, ActULQClampMaxGradVerify);
}  // namespace ge
