/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: op_proto for Operator Clip
 * Author: 
 * Create: 2020-06-12
 */

#include "clip.h"
#include <vector>
#include <string>
#include <iostream>

namespace ge {

    IMPLEMT_VERIFIER(Clip, ClipVerify)
    {
        return GRAPH_SUCCESS;
    }

    IMPLEMT_COMMON_INFERFUNC(ClipInferShape)
    {
        auto shape = op.GetInputDesc("x").GetShape();
        TensorDesc td = op.GetOutputDesc("y");

        td.SetShape(ge::Shape(shape));
        DataType input_dtype = op.GetInputDesc("x").GetDataType();
        td.SetDataType(input_dtype);
        (void)op.UpdateOutputDesc("y", td);
        return GRAPH_SUCCESS;
    }

    COMMON_INFER_FUNC_REG(Clip, ClipInferShape);
    VERIFY_FUNC_REG(Clip, ClipVerify);

}

