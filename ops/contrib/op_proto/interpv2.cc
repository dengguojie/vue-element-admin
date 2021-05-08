/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: interpv2 op proto cpp file
 * Author: huawei
 * Create: 2020-6-17
 */

#include "interpv2.h"
#include <cstdint>
#include <string>
#include <vector>
#include "graph/ge_error_codes.h"
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
    IMPLEMT_VERIFIER(Interpv2, Interpv2Verify)
    {
        DataType inputType0 = op.GetInputDesc(0).GetDataType();
        DataType outputType0 = op.GetOutputDesc(0).GetDataType();
        Format inputFormat = op.GetInputDesc(0).GetFormat();

        if ((inputType0 != DT_FLOAT) && (inputType0 != DT_FLOAT16)) {
            printf("[ERROR][Plugin] Input type %d can't be supported\n", inputType0);
            return GRAPH_FAILED;
        }

        if ((outputType0 != DT_FLOAT) && (outputType0 != DT_FLOAT16)) {
            printf("[ERROR][Plugin] Output type %d can't be supported\n", outputType0);
            return GRAPH_FAILED;
        }

        if (inputFormat != FORMAT_NCHW) {
            printf("[ERROR][Plugin] Input format %d can't be supported\n", inputFormat);
            return GRAPH_FAILED;
        }

        return GRAPH_SUCCESS;
    }

    IMPLEMT_COMMON_INFERFUNC(Interpv2InferShape)
    {
        vector<int64_t> imagesShape = op.GetInputDesc(0).GetShape().GetDims();
        vector<int64_t> input2Shape = op.GetInputDesc(1).GetShape().GetDims();
        TensorDesc td = op.GetOutputDesc("output");
        vector<int64_t> outShape;

        outShape.push_back(imagesShape[0]);
        outShape.push_back(imagesShape[1]);
        outShape.push_back(input2Shape[2]);
        outShape.push_back(input2Shape[3]);
        printf("[INFO][Plugin] Setting Interpv2 output size = {%zu , %zu}\n", outShape[2], outShape[3]);

        td.SetShape(ge::Shape(outShape));
        (void)op.UpdateOutputDesc("output", td);
        return GRAPH_SUCCESS;
    }

    COMMON_INFER_FUNC_REG(Interpv2, Interpv2InferShape);
    VERIFY_FUNC_REG(Interpv2, Interpv2Verify);
}
