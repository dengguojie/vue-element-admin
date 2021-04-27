/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: batch_matmul op proto cpp file
 * Author:
 * Create: 2020-6-11
 * Note:
 */

#include "batch_matmul.h"
#include <vector>
#include <string>

namespace ge {
    IMPLEMT_VERIFIER(BatchMatmul, BatchMatmulVerify) {

        DataType inputTypeOne = op.GetInputDesc(0).GetDataType();
        DataType inputTypeTwo = op.GetInputDesc(1).GetDataType();

        if ((inputTypeOne != DT_FLOAT) && (inputTypeOne != DT_FLOAT16)) {
            printf("[ERROR][Plugin] input_data_left type can't be supported\n");
            return GRAPH_FAILED;
        }
        if ((inputTypeTwo != DT_FLOAT) && (inputTypeTwo != DT_FLOAT16)) {
            printf("[ERROR][Plugin] input_data_right type can't be supported\n");
            return GRAPH_FAILED;
        }

        return GRAPH_SUCCESS;
    }

    IMPLEMT_COMMON_INFERFUNC(BatchMatmulInferShape) {

        auto tensordescInputDataLeft = op.GetInputDesc(0);
        auto tensordescInputDataRight = op.GetInputDesc(1);
        auto tensordescOutput = op.GetOutputDesc(0);
        auto outputShape = tensordescInputDataLeft.GetShape();

        outputShape.SetDim(2, tensordescInputDataRight.GetShape().GetDim(2));

        tensordescOutput.SetShape(outputShape);
        tensordescOutput.SetDataType(tensordescInputDataLeft.GetDataType());

        (void)op.UpdateOutputDesc("output_data", tensordescOutput);
        return GRAPH_SUCCESS;
    }

    COMMON_INFER_FUNC_REG(BatchMatmul, BatchMatmulInferShape);
    VERIFY_FUNC_REG(BatchMatmul, BatchMatmulVerify);
}
