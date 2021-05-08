/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
 * Description: tile op_proto cpp file
 * Author:
 * Create: 2020-6-15
 * Note:
 */

#include "tile.h"
#include <vector>
#include <string>
#include <iostream>

namespace ge {

    IMPLEMT_VERIFIER(TileCaffe, TileCaffeVerify)
    {
        return GRAPH_SUCCESS;
    }

    IMPLEMT_COMMON_INFERFUNC(TileCaffeInferShape)
    {
        auto shape = op.GetInputDesc("x").GetShape();
        TensorDesc td = op.GetOutputDesc("y");

        int32_t axis;
        int32_t tiles;

        op.GetAttr("axis", axis);
        op.GetAttr("tiles", tiles);
        switch (axis)
        {
            case 0:
                shape.SetDim(0, shape.GetDim(0) * tiles);
                break;
            case 1:
                shape.SetDim(1, shape.GetDim(1) * tiles);
                break;
            case 2:
                shape.SetDim(2, shape.GetDim(2) * tiles);
                break;
            case 3:
                shape.SetDim(3, shape.GetDim(3) * tiles);
                break;
            default:
                printf("Error: invalid axis\n");
                return GRAPH_FAILED;
        }
        td.SetShape(ge::Shape(shape));
        DataType input_dtype = op.GetInputDesc("x").GetDataType();
        td.SetDataType(input_dtype);
        (void)op.UpdateOutputDesc("y", td);
        return GRAPH_SUCCESS;
    }

    COMMON_INFER_FUNC_REG(TileCaffe, TileCaffeInferShape);
    VERIFY_FUNC_REG(TileCaffe, TileCaffeVerify);

}

