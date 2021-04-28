/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: l2_normalize op proto cpp file
 * Author:
 * Create: 2020-6-11
 * Note:
 */

#include "l2_normalize.h"
#include <string>
#include <vector>

namespace ge {
    IMPLEMT_VERIFIER(L2Normalize, L2NormalizeVerify)
    {
        return GRAPH_SUCCESS;
    }

    IMPLEMT_COMMON_INFERFUNC(L2NormalizeInferShape)
    {
        auto tensorDesc = op.GetInputDesc("input_data_gm");
        TensorDesc tensordescOutput = op.GetOutputDesc("output_data_gm");

        tensordescOutput.SetShape(tensorDesc.GetShape());
        tensordescOutput.SetDataType(tensorDesc.GetDataType());

        (void)op.UpdateOutputDesc("output_data_gm", tensorDesc);
        return GRAPH_SUCCESS;
    }

    COMMON_INFER_FUNC_REG(L2Normalize, L2NormalizeInferShape);
    VERIFY_FUNC_REG(L2Normalize, L2NormalizeVerify);
}
