/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: faster rcnn first stage processor op proto cpp file
 * Author:
 * Create: 2020-6-17
 */

#include "fasterrcnn_first_stage_processor_tik.h"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "graph/types.h"

namespace ge {
    enum DimNum {
        DIM_BATCH = 0,
        DIM_BOX_NUM = 1,
        DIM_COOR_NUM = 2
    };
    
    IMPLEMT_VERIFIER(FasterrcnnFirstStageProcessorTik, FasterrcnnFirstStageProcessorTikVerify)
    {
        printf("[Plugin][Info] here to verify FasterrcnnFirstStageProcessorTik OP\n");
        if (op.GetInputsSize() != 3) {
            printf("[ERROR][Plugin] number of input must be 3\n");
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }

    IMPLEMT_COMMON_INFERFUNC(FasterrcnnFirstStageProcessorTikInferShape)
    {
        TensorDesc tensordescOutput = op.GetOutputDesc("output_gm");
        vector<vector<int64_t>> match0 { { 1, 29184, 2 }, { 1, 29184, 1, 4 }, { 1, 29184, 1, 4 } };
        vector<vector<int64_t>> match1 { { 1, 61440, 2 }, { 1, 61440, 1, 4 }, { 1, 61440, 1, 4 } };
        for (int i = 0; i < op.GetInputsSize(); i++) {
            if (op.GetInputDesc(i).GetShape().GetDims() != match0[i]) {
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
                (void)op.UpdateOutputDesc("output_gm", tensordescOutput);
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
                (void)op.UpdateOutputDesc("output_gm", tensordescOutput);
                return GRAPH_SUCCESS;
            }
        }
        printf("[ERROR][Plugin]input shape is not valid, please refer to the constraints in header file\n");

        return GRAPH_FAILED;
    }
    // Registered inferfunction
    COMMON_INFER_FUNC_REG(FasterrcnnFirstStageProcessorTik, FasterrcnnFirstStageProcessorTikInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(FasterrcnnFirstStageProcessorTik, FasterrcnnFirstStageProcessorTikVerify);
}
