/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: rank op proto cpp file
 * Author:
 * Create: 2020-6-17
 */
#include "tik_rank.h"
#include <string>
#include <vector>

namespace ge {
    IMPLEMT_VERIFIER(TikRank, RankVerify) {
        if (op.GetInputsSize() != 1) {
            printf("[ERROR][Plugin] number of input must be 1\n");
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }

    IMPLEMT_COMMON_INFERFUNC(RankInferShape) {
        TensorDesc tensordescOutput = op.GetOutputDesc("output_gm");
        vector<int64_t> dimVec;
        dimVec.push_back(1);
        ge::Shape outputShape = ge::Shape(dimVec);
        tensordescOutput.SetShape(outputShape);
        tensordescOutput.SetDataType(DT_INT32);
        tensordescOutput.SetFormat(FORMAT_ND);
        (void)op.UpdateOutputDesc("output_gm", tensordescOutput);
        return GRAPH_SUCCESS;
    }

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(TikRank, RankInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(TikRank, RankVerify);
}
