/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: op_proto for Operator ImagePad
 * Author: huawei
 * Create: 2020-06-11
 */

#include "image_pad.h"
#include "graph/ge_error_codes.h"
#include "graph/operator_reg.h"

namespace ge {
    IMPLEMT_VERIFIER(ImagePad, ImagePadVerity)
    {
        printf("[Plugin][Info] here to verify ImagePad OP\n");
        DataType inputTypeZero = op.GetInputDesc(0).GetDataType();
        DataType outputTypeZero = op.GetOutputDesc(0).GetDataType();
        Format inputFormat = op.GetInputDesc("input_dict").GetFormat();

        vector<int64_t> vImageShape = op.GetInputDesc("input_dict").GetShape().GetDims();
        if (vImageShape.size() != 3) {
            printf("[ERROR][Plugin] input shape only supporte 3D\n");
            return GRAPH_FAILED;
        }
        
        if (inputTypeZero != DT_FLOAT16 && inputTypeZero != DT_FLOAT) {
            printf("[ERROR][Plugin] input type %d can't be supported\n", inputTypeZero);
            return GRAPH_FAILED;
        }

        if (outputTypeZero != DT_FLOAT16 && outputTypeZero != DT_FLOAT) {
            printf("[ERROR][Plugin] Output type %d can't be supported\n", outputTypeZero);
            return GRAPH_FAILED;
        }

        return GRAPH_SUCCESS;
    }

    IMPLEMT_COMMON_INFERFUNC(ImagePadInferShape)
    {
        printf("[Plugin][Info] here to infer shape ImagePad Op\n");

        vector<vector<int64_t>> paddings;
        if (op.GetAttr("paddings", paddings) != GRAPH_SUCCESS)
        {
            printf("Get paddings failed!\n");
            return GRAPH_FAILED;
        }

        TensorDesc tensorDescOutput = op.GetOutputDesc(0);
        TensorDesc tensorDescInput = op.GetInputDesc(0);
        tensorDescInput.SetFormat(FORMAT_ND);
        auto shape = tensorDescInput.GetShape();

        shape.SetDim(0, shape.GetDim(0) + paddings[0][0] + paddings[0][1]);
        shape.SetDim(1, shape.GetDim(1) + paddings[1][0] + paddings[1][1]);
        shape.SetDim(2, shape.GetDim(2) + paddings[2][0] + paddings[2][1]);

        tensorDescOutput.SetShape(shape);
        tensorDescOutput.SetDataType(tensorDescInput.GetDataType());
        tensorDescOutput.SetFormat(FORMAT_ND);

        (void)op.UpdateOutputDesc("input_dict", tensorDescOutput);
        (void)op.UpdateOutputDesc("output_dict", tensorDescOutput);

        return GRAPH_SUCCESS;
    }

    COMMON_INFER_FUNC_REG(ImagePad, ImagePadInferShape);
    VERIFY_FUNC_REG(ImagePad, ImagePadVerity);
}
