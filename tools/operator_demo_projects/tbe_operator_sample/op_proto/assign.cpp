/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Description: Huawei Code
 *
 * Author: Huawei
 *
 * Create: 2020-01-01
 *
 */
#include "./assign.h"
#include <string>
#include <vector>

namespace ge {
static bool CheckTwoInputDtypeSame(const Operator &op, const string &inputName1, const string &inputName2)
{
    DataType inputTypeX1 = op.GetInputDescByName(inputName1.c_str()).GetDataType();
    DataType inputTypeX2 = op.GetInputDescByName(inputName2.c_str()).GetDataType();
    if (inputTypeX1 != inputTypeX2) {
        return false;
    }

    return true;
}

IMPLEMT_VERIFIER(Assign, AssignVerify)
{
    if (!CheckTwoInputDtypeSame(op, "ref", "value")) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AssignInferShape)
{
    TensorDesc tensordesc_output = op.GetOutputDescByName("ref");

    tensordesc_output.SetShape(op.GetInputDescByName("ref").GetShape());
    tensordesc_output.SetDataType(op.GetInputDescByName("ref").GetDataType());

    (void)op.UpdateOutputDesc("ref", tensordesc_output);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Assign, AssignInferShape);
VERIFY_FUNC_REG(Assign, AssignVerify);

} // namespace ge
