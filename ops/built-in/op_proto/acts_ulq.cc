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


namespace ge {

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ActsULQInferShape) {
    Shape x_shape = op.GetInputDesc("x").GetShape();
    Shape clamp_min_shape = op.GetInputDesc("clamp_min").GetShape();
    Shape clamp_max_shape = op.GetInputDesc("clamp_max").GetShape();

    if (clamp_min_shape.GetShapeSize() != 1) {
        OP_LOGE(TbeGetName(op).c_str(), "The size of clamp_min must be 1!");
        return GRAPH_FAILED;
    }

    if (clamp_max_shape.GetShapeSize() != 1) {
        OP_LOGE(TbeGetName(op).c_str(), "The size of clamp_max must be 1!");
        return GRAPH_FAILED;
    }

    TensorDesc y = op.GetOutputDesc("y");
    y.SetShape(x_shape);
    y.SetDataType(op.GetInputDesc("x").GetDataType());
    if (op.UpdateOutputDesc("y", y) != GRAPH_SUCCESS) {
        OP_LOGE(TbeGetName(op).c_str(), "Update output[y] failed!");
        return GRAPH_FAILED;
    }

    TensorDesc clamp_min_mask = op.GetOutputDesc("clamp_min_mask");
    clamp_min_mask.SetShape(x_shape);
    if (op.UpdateOutputDesc("clamp_min_mask", clamp_min_mask) != GRAPH_SUCCESS) {
        OP_LOGE(TbeGetName(op).c_str(), "Update output[clamp_min_mask] failed!");
        return GRAPH_FAILED;
    }

    TensorDesc clamp_max_mask = op.GetOutputDesc("clamp_max_mask");
    clamp_max_mask.SetShape(x_shape);
    if (op.UpdateOutputDesc("clamp_max_mask", clamp_max_mask) != GRAPH_SUCCESS) {
        OP_LOGE(TbeGetName(op).c_str(), "Update output[clamp_max_mask] failed!");
        return GRAPH_FAILED;
    }

    TensorDesc x_clamped_loss = op.GetOutputDesc("x_clamped_loss");
    x_clamped_loss.SetShape(x_shape);
    x_clamped_loss.SetDataType(op.GetInputDesc("x").GetDataType());
    if (op.UpdateOutputDesc("x_clamped_loss", x_clamped_loss) != GRAPH_SUCCESS) {
        OP_LOGE(TbeGetName(op).c_str(), "Update output[x_clamped_loss] failed!");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ActsULQ, ActsULQVerify) {
    DataType x_type = op.GetInputDesc("x").GetDataType();
    DataType clamp_min_type = op.GetInputDesc("clamp_min").GetDataType();
    DataType clamp_max_type = op.GetInputDesc("clamp_max").GetDataType();

    if (x_type != clamp_min_type) {
        OP_LOGE(TbeGetName(op).c_str(), "The type of clamp_min must be the same as x!");
        return GRAPH_FAILED;
    }

    if (x_type != clamp_max_type) {
        OP_LOGE(TbeGetName(op).c_str(), "The type of clamp_max must be the same as x!");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(ActsULQ, ActsULQInferShape);

// Registered verify function
VERIFY_FUNC_REG(ActsULQ, ActsULQVerify);
}  // namespace ge
