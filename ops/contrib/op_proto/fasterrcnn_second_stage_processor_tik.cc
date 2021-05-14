/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: faster rcnn second stage processor op proto cpp file
 * Author: 
 * Create: 2020-6-17
 */

#include "fasterrcnn_second_stage_processor_tik.h"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "graph/types.h"
using namespace std;

namespace ge {
    IMPLEMT_VERIFIER(FasterrcnnSecondStageProcessorTik, FasterrcnnSecondStageProcessorTikVerify)
    {
        printf("[Plugin][Info] here to verify FasterrcnnSecondStageProcessorTik OP\n");
        if (op.GetInputsSize() != 3) {
            printf("[ERROR][Plugin] number of input must be 3\n");
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }

    IMPLEMT_COMMON_INFERFUNC(FasterrcnnSecondStageProcessorTikInferShape)
    {
        string outName[5] = {"score_out_gm","output_class", "output_boxes", \
            "output_num_detection", "output_score"};
        vector<vector<int64_t>> match { { 100, 360 }, { 1, 100, 4 }, { 100, 91 } };  // support shape1
        vector<vector<int64_t>> match1 { { 300, 8 }, { 1, 300, 4 }, { 300, 3 } };    // support shape2 
        vector<vector<int64_t>> outputSet { { 1, 100, 8 }, { 100 }, { 1, 100, 4 }, { 1 }, { 100 } };
        for (int i = 0; i < op.GetInputsSize(); i++) {
            vector<int64_t> dims = op.GetInputDesc(i).GetShape().GetDims();
            if (dims != match[i] && dims != match1[i]) {
                break;
            }
            if (i == op.GetInputsSize() - 1) {
                for (unsigned int j = 0; j < op.GetOutputsSize(); j++) {
                    TensorDesc tensordescOutput = op.GetOutputDesc(j);
                    vector<int64_t> outputShape(outputSet[j]);
                    tensordescOutput.SetDataType(op.GetInputDesc(0).GetDataType());
                    tensordescOutput.SetShape((Shape)outputShape);
                    tensordescOutput.SetFormat(FORMAT_ND);
                    (void)op.UpdateOutputDesc(outName[j], tensordescOutput);
                }
                return GRAPH_SUCCESS;
            }
        }

        printf("[ERROR][Plugin]input shape is not valid, please refer to the \
            constraints in header file\n");
        return GRAPH_FAILED;
    }

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(FasterrcnnSecondStageProcessorTik, FasterrcnnSecondStageProcessorTikInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(FasterrcnnSecondStageProcessorTik, FasterrcnnSecondStageProcessorTikVerify);
}
