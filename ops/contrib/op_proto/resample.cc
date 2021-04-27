/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "resample.h"
#include <vector>
#include <string>
#include <iostream>

namespace ge {

    enum ResampleType {
        NEAREST = 1,
        LINEAR = 2,
        CUBIC = 3,
        AREA = 4,
    };

    IMPLEMT_VERIFIER(Resample, ResampleVerify)
    {
        return GRAPH_SUCCESS;
    }

    IMPLEMT_COMMON_INFERFUNC(ResampleInferShape)
    {
        auto shape = op.GetInputDesc("x").GetShape();
        TensorDesc td = op.GetOutputDesc("y");
        int64_t height, width;

        op.GetAttr("height", height);
        op.GetAttr("width", width);

        shape.SetDim(2, height);
        shape.SetDim(3, width);
        td.SetShape(ge::Shape(shape));
        DataType input_dtype = op.GetInputDesc("x").GetDataType();
        td.SetDataType(input_dtype);
        (void)op.UpdateOutputDesc("y", td);
        return GRAPH_SUCCESS;
    }

    COMMON_INFER_FUNC_REG(Resample, ResampleInferShape);
    VERIFY_FUNC_REG(Resample, ResampleVerify);

}

