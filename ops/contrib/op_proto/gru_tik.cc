/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: gru op proto cpp file
 * Author:
 * Create: 2020-6-17
 */

#include "gru_tik.h"

namespace ge {
    IMPLEMT_VERIFIER(GruTik, GruTikVerify)
    {
        vector<int64_t> inputShape = op.GetInputDesc("input").GetShape().GetDims();
        DataType inputType0 = op.GetInputDesc(0).GetDataType();
        DataType outputType0 = op.GetOutputDesc(0).GetDataType();

        if (inputShape.size() != 3) {
            printf("[ERROR][Plugin] input shape only supporte 3D\n");
            return GRAPH_FAILED;
        }

        if (inputType0 != DT_FLOAT && inputType0 != DT_FLOAT16) {
            printf("[ERROR][Plugin] Input type %d can't be supported\n", inputType0);
            return GRAPH_FAILED;
        }

        if (outputType0 != DT_FLOAT && inputType0 != DT_FLOAT16) {
            printf("[ERROR][Plugin] Output type %d can't be supported\n", outputType0);
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }

    IMPLEMT_COMMON_INFERFUNC(GruTikInferShape)
    {
        printf("[Plugin][Info] GruTik infer shape begin.\n");
        TensorDesc td = op.GetOutputDesc("output");
        auto output_shape = op.GetInputDesc("input").GetShape();
        int units = 0;
        if (GRAPH_SUCCESS != op.GetAttr("units", units)) {
            printf("Get attr units failed!\n");
            return GRAPH_FAILED;
        }

        bool fusegru = true;
        if (GRAPH_SUCCESS != op.GetAttr("fusegru", fusegru)) {
            printf("Get attr fusegru failed!\n");
            return GRAPH_FAILED;
        }
        output_shape.SetDim(2, units * (1 + int(fusegru)));
        td.SetShape(ge::Shape(output_shape));
        (void)op.UpdateOutputDesc("output", td);
        return GRAPH_SUCCESS;
    }
    COMMON_INFER_FUNC_REG(GruTik, GruTikInferShape);
    VERIFY_FUNC_REG(GruTik, GruTikVerify);
}
