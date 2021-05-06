/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: WarpPerspective caffe opp header file
 * Author: Huawei
 * Create: 2020-6-11
 */

#include "warp_perspective.h"
#include "graph/ge_error_codes.h"
#include "graph/operator_reg.h"

namespace ge {
IMPLEMT_VERIFIER(WarpPerspective, WarpPerspectiveVerity)
{
    printf("[Plugin][Info] here to verify WarpPerspective Op\n");
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(WarpPerspectiveInferShape)
{
    printf("[Plugin][Info] here to infer shape WarpPerspective Op\n");
    TensorDesc tensordescOutput = op.GetOutputDesc("y");
    TensorDesc tensordescInput = op.GetInputDesc("x");
    auto shapeDesc = tensordescInput.GetShape();

    int dstHeight;
    int dstWidth;

    if (GRAPH_SUCCESS != op.GetAttr("dst_height", dstHeight)) {
        printf("Get dst_height failed!\n");
        return GRAPH_FAILED;
    }

    if (GRAPH_SUCCESS != op.GetAttr("dst_width", dstWidth)) {
        printf("Get dst_width failed!\n");
        return GRAPH_FAILED;
    }
    printf("[Plugin][Info] Get dst_height: %d\n", dstHeight);
    printf("[Plugin][Info] Get dst_width: %d\n", dstWidth);
    shapeDesc.SetDim(2, dstHeight);
    shapeDesc.SetDim(3, dstWidth);
    tensordescOutput.SetShape(shapeDesc);
    tensordescOutput.SetDataType(tensordescInput.GetDataType());
    tensordescOutput.SetFormat(tensordescInput.GetFormat());

    (void)op.UpdateOutputDesc("y", tensordescOutput);

    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(WarpPerspective, WarpPerspectiveInferShape);
VERIFY_FUNC_REG(WarpPerspective, WarpPerspectiveVerity);
}
