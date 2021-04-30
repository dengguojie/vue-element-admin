/**
Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
Description: op_proto for interp caffe operator
Author:
Create: 2020-6-11
*/
#include "interp.h"
#include <iostream>
#include <vector>
#include <string>


const int64_t DEFAULT_ZOOM_FACTOR = 1;
const int64_t DEFAULT_SHRINK_FACTOR = 1;
const int64_t DEFAULT_HEIGHT = 0;
const int64_t DEFAULT_WIDTH = 0;
const int64_t DEFAULT_PAD_BEG = 0;
const int64_t DEFAULT_PAD_END = 0;

struct InterpParam {
    int64_t shrinkFactor;
    int64_t zoomFactor;
    int64_t height;
    int64_t width;
};

void CalculateOutShape(const struct InterpParam param,  const int64_t (&inShape)[2], int64_t (&outShape)[2])
{
    // only support pad = 0 now
    int64_t padBeg = DEFAULT_PAD_BEG;
    int64_t padEnd = DEFAULT_PAD_END;
    int64_t heightInEff = inShape[0] + padBeg + padEnd;
    int64_t widthInEff = inShape[1] + padBeg + padEnd;

    if ((param.shrinkFactor != DEFAULT_SHRINK_FACTOR) && (param.zoomFactor == DEFAULT_ZOOM_FACTOR)) {
        if (param.shrinkFactor == 0) {
            return;
        }
        outShape[0] = (heightInEff - 1) / param.shrinkFactor + 1;
        outShape[1] = (widthInEff - 1) / param.shrinkFactor + 1;
    } else if ((param.shrinkFactor == DEFAULT_SHRINK_FACTOR) && (param.zoomFactor != DEFAULT_ZOOM_FACTOR)) {
        outShape[0] = heightInEff + (heightInEff - 1) * (param.zoomFactor - 1);
        outShape[1] = widthInEff + (widthInEff - 1) * (param.zoomFactor - 1);
    } else if ((param.height != DEFAULT_HEIGHT) && (param.width != DEFAULT_WIDTH)) {
        outShape[0] = param.height;
        outShape[1] = param.width;
    } else if ((param.shrinkFactor != DEFAULT_SHRINK_FACTOR) && (param.zoomFactor != DEFAULT_ZOOM_FACTOR)) {
        if (param.shrinkFactor == 0) {
            return;
        }
        outShape[0] = (heightInEff - 1) / param.shrinkFactor + 1;
        outShape[1] = (widthInEff - 1) / param.shrinkFactor + 1;
        outShape[0] = outShape[0] + (outShape[0] - 1) * (param.zoomFactor - 1);
        outShape[1] = outShape[1] + (outShape[1] - 1) * (param.zoomFactor - 1);
    }
}

namespace ge {
IMPLEMT_VERIFIER(Interp, InterpVerify)
{
    DataType inputType0 = op.GetInputDesc(0).GetDataType();
    DataType outputType0 = op.GetOutputDesc(0).GetDataType();
    Format inputFormat = op.GetInputDesc("images").GetFormat();

    if ((inputType0 != DT_FLOAT) && (inputType0 != DT_FLOAT16)) {
        printf("[ERROR][Plugin] Output type %d can't be supported\n", inputType0);
        return GRAPH_FAILED;
    }

    if ((outputType0 != DT_FLOAT) && (outputType0 != DT_FLOAT16)) {
        printf("[ERROR][Plugin] Output type %d can't be supported\n", outputType0);
        return GRAPH_FAILED;
    }

    if ((inputFormat != FORMAT_NHWC) && (inputFormat != FORMAT_NCHW)) {
        printf("[ERROR][Plugin] Input format %d can't be supported\n", inputFormat);
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(InterpInferShape)
{
    vector<int64_t> imagesShape = op.GetInputDesc("images").GetShape().GetDims();

    int64_t inShape[2] = {0};

    Format inputFormat = op.GetInputDesc("images").GetFormat();
    if (inputFormat == FORMAT_NHWC) {
        inShape[0] = imagesShape[1];
        inShape[1] = imagesShape[2];
    } else if (inputFormat == FORMAT_NCHW) {
        inShape[0] = imagesShape[2];
        inShape[1] = imagesShape[3];
    }

    printf("[INFO][Plugin] Setting Interp input size = {%zu, %zu}\n", inShape[0], inShape[1]);

    struct InterpParam interpParam = {0, 0, 0, 0};

    op.GetAttr("shrink_factor", interpParam.shrinkFactor);
    op.GetAttr("zoom_factor", interpParam.zoomFactor);
    op.GetAttr("height", interpParam.height);
    op.GetAttr("width", interpParam.width);

    // default out H/W == in H/W
    int64_t outShape[2] = {inShape[0], inShape[1]};
    CalculateOutShape(interpParam, inShape, outShape);

    TensorDesc td = op.GetOutputDesc("y");
    vector<int64_t> yShape;
    if (inputFormat == FORMAT_NHWC) {
        yShape.push_back(imagesShape[0]);
        yShape.push_back(outShape[0]);
        yShape.push_back(outShape[1]);
        yShape.push_back(imagesShape[3]);
    } else if (inputFormat == FORMAT_NCHW) {
        yShape.push_back(imagesShape[0]);
        yShape.push_back(imagesShape[1]);
        yShape.push_back(outShape[0]);
        yShape.push_back(outShape[1]);
    }

    op.SetAttr("height", outShape[0]);
    op.SetAttr("width", outShape[1]);
    printf("[INFO][Plugin] Setting Interp output size = {%zu, %zu}\n", outShape[0], outShape[1]);

    td.SetShape(ge::Shape(yShape));
    td.SetDataType(DT_FLOAT);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Interp, InterpInferShape);
VERIFY_FUNC_REG(Interp, InterpVerify);
}
