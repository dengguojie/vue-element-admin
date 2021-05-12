/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: op_proto for Operator Heatmap2Coord
 * Author: Huawei
 * Create: 2020-06-12
 */

#include "heatmap2coord.h"
#include <vector>
#include <string>
#include <iostream>

namespace ge {

IMPLEMT_VERIFIER(Heatmap2Coord, Heatmap2coordVerify) 
{
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(Heatmap2coordInferShape) 
{
    vector<int64_t> x_shape = op.GetInputDesc("x").GetShape().GetDims();

    TensorDesc td = op.GetOutputDesc("y");
    vector<int64_t> y_shape;
    y_shape.push_back(x_shape[0]);
    y_shape.push_back(x_shape[1]);
    y_shape.push_back(1);
    y_shape.push_back(2);

    td.SetShape(ge::Shape(y_shape));
    td.SetDataType(DT_FLOAT16);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Heatmap2Coord, Heatmap2coordInferShape);
VERIFY_FUNC_REG(Heatmap2Coord, Heatmap2coordVerify);

}
