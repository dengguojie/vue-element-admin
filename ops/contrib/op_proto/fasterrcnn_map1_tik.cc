/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: faster rcnn map1 op proto cpp file
 * Author:
 * Create: 2020-6-17
 */

#include "fasterrcnn_map1_tik.h"
#include <string>
#include <vector>

namespace ge {
    enum DimNum {
        DIM_BATCH = 0,
        DIM_BOX_NUM = 1,
        DIM_COOR_NUM = 2
    };
    
    IMPLEMT_VERIFIER(FasterrcnnMap1Tik, FasterrcnnMap1TikVerify)
    {
        printf("[Plugin][Info] here to verify FasterrcnnMap1Tik OP\n");
        if (op.GetInputsSize() != 1) {
            printf("[ERROR][Plugin] number of input must be 1\n");
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }

    IMPLEMT_COMMON_INFERFUNC(FasterrcnnMap1TikInferShape)
    {
        TensorDesc tensordescOutput = op.GetOutputDesc(0);
        vector<vector<int64_t>> match { { 1, 100, 4 } };
        vector<vector<int64_t>> match1 { { 1, 300, 4 } };
        for (int i = 0; i < op.GetInputsSize(); i++) {
            if (op.GetInputDesc(i).GetShape().GetDims() != match[i]) {
                break;
            }
            if (i == op.GetInputsSize() - 1) {
                vector<int64_t> outputShape(3, 0);
                outputShape[DIM_BATCH] = 1;
                outputShape[DIM_BOX_NUM] = 100;
                outputShape[DIM_COOR_NUM] = 4;
                tensordescOutput.SetShape((Shape)outputShape);
                tensordescOutput.SetDataType(op.GetInputDesc(0).GetDataType());
                tensordescOutput.SetFormat(FORMAT_ND);
                (void)op.UpdateOutputDesc("proposal_abso_gm", tensordescOutput);
                return GRAPH_SUCCESS;
            }
        }
        for (int i = 0; i < op.GetInputsSize(); i++) {
            if (op.GetInputDesc(i).GetShape().GetDims() != match1[i]) {
                break;
            }
            if (i == op.GetInputsSize() - 1) {
                vector<int64_t> outputShape(3, 0);
                outputShape[DIM_BATCH] = 1;
                outputShape[DIM_BOX_NUM] = 300;
                outputShape[DIM_COOR_NUM] = 4;
                tensordescOutput.SetShape((Shape)outputShape);
                tensordescOutput.SetDataType(op.GetInputDesc(0).GetDataType());
                tensordescOutput.SetFormat(FORMAT_ND);
                (void)op.UpdateOutputDesc("proposal_abso_gm", tensordescOutput);
                return GRAPH_SUCCESS;
            }
        }

        printf("[ERROR][Plugin]input shape is not valid, \
            please refer to the constraints in header file\n");
        return GRAPH_FAILED;
    }
    // Registered inferfunction
    COMMON_INFER_FUNC_REG(FasterrcnnMap1Tik, FasterrcnnMap1TikInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(FasterrcnnMap1Tik, FasterrcnnMap1TikVerify);
}
