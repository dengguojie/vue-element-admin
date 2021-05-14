/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: faster rcnn preprocessor op proto cpp file
 * Author: 
 * Create: 2020-6-17
 */

#include "fasterrcnn_preprocessor_tik.h"
#include <string>
#include <vector>

enum DimNum {
    DIM_N = 0,
    DIM_H = 1,
    DIM_W = 2,
    DIM_C = 3
};

namespace ge {
    IMPLEMT_VERIFIER(FasterrcnnPreprocessorTik, FasterrcnnPreprocessorTikVerify)
    {
        printf("[Plugin][Info] here to verify FasterrcnnPreprocessorTik OP\n");
        if (op.GetInputsSize() != 1) {
            printf("[ERROR][Plugin] number of input must be 1\n");
            return GRAPH_FAILED;
        }

        return GRAPH_SUCCESS;
    }

    IMPLEMT_COMMON_INFERFUNC(FasterrcnnPreprocessorTikInferShape)
    {
        TensorDesc tensordescOutput = op.GetOutputDesc(0);
        vector<vector<int64_t>> match { { 1, 300, 960, 3 } };
        for (int i = 0; i < op.GetInputsSize(); i++) {
            if (op.GetInputDesc(i).GetShape().GetDims() != match[i]) {
                break;
            }
            if (i == op.GetInputsSize() - 1) {
                vector<int64_t> outputShape(4, 0);
                outputShape[DIM_N] = 1;
                outputShape[DIM_H] = 320;
                outputShape[DIM_W] = 1024;
                outputShape[DIM_C] = 3;
                tensordescOutput.SetShape((Shape)outputShape);
                tensordescOutput.SetDataType(DT_FLOAT);
                tensordescOutput.SetFormat(FORMAT_NHWC);
                (void)op.UpdateOutputDesc("dst_gm", tensordescOutput);
                return GRAPH_SUCCESS;
            }
        }
        printf("[ERROR][Plugin]input shape is not valid, please refer to the \
                constraints in header file\n");
        return GRAPH_FAILED;
    }

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(FasterrcnnPreprocessorTik, FasterrcnnPreprocessorTikInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(FasterrcnnPreprocessorTik, FasterrcnnPreprocessorTikVerify);
} 
