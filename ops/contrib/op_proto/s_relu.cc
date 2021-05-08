/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: srelu op proto cpp file
 * Author: Huawei
 * Create: 2020-06-12
 */
#include "s_relu.h"
#include <iostream>
#include <string>
#include <vector>

namespace ge {
IMPLEMT_VERIFIER(SReLU, SreluVerify)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SreluInferShape)
{
    std::vector<int64_t> vecShape = op.GetInputDesc("x").GetShape().GetDims();
    TensorDesc tensorDesc = op.GetOutputDesc("y");

    tensorDesc.SetShape(ge::Shape(vecShape));
    DataType inputType = op.GetInputDesc("x").GetDataType();
    tensorDesc.SetDataType(inputType);
    (void)op.UpdateOutputDesc("y", tensorDesc);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SReLU, SreluInferShape);
VERIFY_FUNC_REG(SReLU, SreluVerify);
}
