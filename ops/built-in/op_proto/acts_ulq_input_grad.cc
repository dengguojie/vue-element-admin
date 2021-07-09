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
 * \file acts_ulq_input_grad.cpp
 * \brief
 */
#include <vector>
#include <string>

#include "math_ops.h"
#include "op_log.h"


namespace ge {

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ActsULQInputGradInferShape) {
    Shape y_grad_shape = op.GetInputDesc("y_grad").GetShape();
    Shape clamp_min_mask_shape = op.GetInputDesc("clamp_min_mask").GetShape();
    Shape clamp_max_mask_shape = op.GetInputDesc("clamp_max_mask").GetShape();

    if (y_grad_shape.GetDims() != clamp_min_mask_shape.GetDims()) {
        OP_LOGE(op.GetName().c_str(), "The shape of clamp_min_mask must be the same as y_grad!");
        return GRAPH_FAILED;
    }

    if (y_grad_shape.GetDims() != clamp_max_mask_shape.GetDims()) {
        OP_LOGE(op.GetName().c_str(), "The shape of clamp_max_mask must be the same as y_grad!");
        return GRAPH_FAILED;
    }

    TensorDesc x_grad = op.GetOutputDesc("x_grad");
    x_grad.SetShape(y_grad_shape);
    x_grad.SetDataType(op.GetInputDesc("y_grad").GetDataType());
    if (op.UpdateOutputDesc("x_grad", x_grad) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Update output[x_grad] failed!");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ActsULQInputGrad, ActsULQInputGradVerify) {
    DataType clamp_min_mask_type = op.GetInputDesc("clamp_min_mask").GetDataType();
    DataType clamp_max_mask_type = op.GetInputDesc("clamp_max_mask").GetDataType();

    if (clamp_min_mask_type != clamp_max_mask_type) {
        OP_LOGE(op.GetName().c_str(), "The type of clamp_min_mask must be the same as clamp_max_mask!");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(ActsULQInputGrad, ActsULQInputGradInferShape);

// Registered verify function
VERIFY_FUNC_REG(ActsULQInputGrad, ActsULQInputGradVerify);
}  // namespace ge
