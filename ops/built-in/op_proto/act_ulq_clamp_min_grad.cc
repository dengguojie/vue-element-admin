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

IMPLEMT_VERIFIER(ActULQClampMinGrad, ActULQClampMinGradVerify) {
    DataType y_grad = op.GetInputDesc("y_grad").GetDataType();
    DataType clamp_min_mask = op.GetInputDesc("clamp_min_mask").GetDataType();
    DataType x_clamped_loss = op.GetInputDesc("x_clamped_loss").GetDataType();

    if (clamp_min_mask != ge::DT_BOOL && clamp_min_mask != ge::DT_INT8) {
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ActULQClampMinGradInferShape) {
    Shape y_grad_shape = op.GetInputDesc("y_grad").GetShape();
    TensorDesc clamp_min_grad = op.GetOutputDesc("clamp_min_grad");
    clamp_min_grad.SetShape({});
    clamp_min_grad.SetDataType(ge::DT_FLOAT);
    (void)op.UpdateOutputDesc("clamp_min_grad", clamp_min_grad);

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(ActULQClampMinGrad, ActULQClampMinGradInferShape);

// Registered verify function
VERIFY_FUNC_REG(ActULQClampMinGrad, ActULQClampMinGradVerify);
}  // namespace ge
