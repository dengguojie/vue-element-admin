/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: CTCGreedyDecoder op proto cpp file
 * Author: HUawei
 * Create: 2020-6-17
 */


#include "ctc_greedy_decoder.h"

namespace ge {
    IMPLEMT_VERIFIER(ctc_greedy_decoder, ctc_greedy_decoderVerify)
    {
        vector<int64_t> inputShape = op.GetInputDesc("input").GetShape().GetDims();
        DataType inputType = op.GetInputDesc(0).GetDataType();
        DataType outputType = op.GetOutputDesc(0).GetDataType();
        Format inputFormat = op.GetInputDesc("input").GetFormat();

        if (inputShape.size() != 3) {
                printf("[ERROR][Plugin] input shape only supporte 3D\n");
                return GRAPH_FAILED;
        }

        if (inputType != DT_FLOAT && inputType != DT_FLOAT16) {
            printf("[ERROR][Plugin] Input type %d can't be supported\n", inputType);
            return GRAPH_FAILED;
        }

        if (outputType != DT_FLOAT && inputType != DT_FLOAT16) {
            printf("[ERROR][Plugin] Output type %d can't be supported\n", outputType);
            return GRAPH_FAILED;
        }

        return GRAPH_SUCCESS;
    }

    IMPLEMT_COMMON_INFERFUNC(ctc_greedy_decoderInferShape)
    {
        printf("[Plugin][Info] CTCGreedyDecoder infer shape begin.\n");
        TensorDesc td = op.GetOutputDesc("output");
        auto outputShape = op.GetInputDesc("input").GetShape();
        vector<int64_t> sequenceLength;
        if (GRAPH_SUCCESS != op.GetAttr("sequence_length", sequenceLength)) {
            printf("Get sequence_length failed!\n");
            return GRAPH_FAILED;
        }

        int outputLen = 0;
        for (int i = 0; i < sequenceLength.size(); i++) {
            outputLen = sequenceLength[i] > outputLen ? sequenceLength[i] : outputLen;
            printf("sequence_length_%d = %d\n", i, outputLen);
        }

        printf("CTCGreedyDecoder output len: %d\n", outputLen);

        outputShape.SetDim(0, outputShape.GetDim(0));
        outputShape.SetDim(1, outputLen + 1);
        outputShape.SetDim(2, 1);
        bool mergeRepeated = false;
        int defaultValue = -1;
        op.GetAttr("merge_repeated", mergeRepeated);
        op.GetAttr("default_value", defaultValue);
        td.SetShape(ge::Shape(outputShape));
        (void)op.UpdateOutputDesc("output", td);
        return GRAPH_SUCCESS;
    }
    COMMON_INFER_FUNC_REG(ctc_greedy_decoder, ctc_greedy_decoderInferShape);
    VERIFY_FUNC_REG(ctc_greedy_decoder, ctc_greedy_decoderVerify);
}

