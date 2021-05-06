/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: YoloV3 postprocessor op proto cpp file
 * Author:
 * Create: 2020-6-11
 * Note:
 */
#include "yolov3_post_processor.h"
#include <vector>
#include <string>

namespace ge {
IMPLEMT_VERIFIER(Yolov3PostProcessor, Yolov3PostProcessorVerify)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(Yolov3PostProcessorInferShape)
{
    TensorDesc tensorDescOutput = op.GetOutputDesc("output_gm");

    vector<int64_t> dimVec;
    dimVec.push_back(100);
    dimVec.push_back(8);
    ge::Shape outputShape = ge::Shape(dimVec);
    tensorDescOutput.SetShape(outputShape);

    (void)op.UpdateOutputDesc("output_gm", tensorDescOutput);
    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(Yolov3PostProcessor, Yolov3PostProcessorInferShape);

// Registered verify function
VERIFY_FUNC_REG(Yolov3PostProcessor, Yolov3PostProcessorVerify);
}
