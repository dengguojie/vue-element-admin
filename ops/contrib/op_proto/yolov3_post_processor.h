/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file yolov3_post_processor.h
 *
 * @version 1.0
 */

#ifndef GE_OP_TIK_H
#define GE_OP_TIK_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief TF yolov3 post processor fusion operator.
 *
 * @par Inputs:
 * 11 inputs, including:
 * @li reshape_1_gm, yolov3 featuremap1, datatype:fp16, format:ND, shape:169x288
 * @li reshape_7_gm, yolov3 featuremap2, datatype:fp16, format:ND, shape:676x288
 * @li reshape_13_gm, yolov3 featuremap3, datatype:fp16, format:ND, shape:2704x288
 * @li anchor13_gm, featuremap1's anchors, datatype:fp16, format:ND, shape:2x512
 * @li anchor26_gm, featuremap2's anchors, datatype:fp16, format:ND, shape:8x512
 * @li anchor52_gm, featuremap3's anchors, datatype:fp16, format:ND, shape:32x512
 * @li grid13_gm, featuremap1's grids, datatype:fp16, format:ND, shape:2x512
 * @li grid26_gm, featuremap2's grids, datatype:fp16, format:ND, shape:2x512
 * @li gird52_gm, featuremap3's grids, datatype:fp16, format:ND, shape:2x512
 * @li mask_gm, mask for accelerated computing, datatype:fp16, format:ND, shape:512
 * @li objclass_gm, 80's coco labels for accelerated computing, datatype:fp16, format:ND, shape:80x128
 *
 * @par Attributes:
 *
 * @par Outputs:
 * 1 output, including:
 * @li output_gm, TF yolov3 output result, The 100 x 8 matrix is used for description.
 * A maximum of 100 rows are supported. The eight columns are [y value in the upper left corner,
 * x value in the upper left corner, y value in the lower right corner, x value in the lower right corner,
 * score, category, reserved, reserved]. datatype:fp16, format:ND, shape:512
 *
 * @attention Constraints:
 * This operator is a fusion operator and supports only COCO 80-class object detection.
 * It can be used only after the TF Yolov3 network model is modified.
 */

REG_OP(Yolov3PostProcessor)
    .INPUT(reshape_1_gm, TensorType({ DT_FLOAT16 }))  /* "1st operand." */
    .INPUT(reshape_7_gm, TensorType({ DT_FLOAT16 }))  /* "2st operand." */
    .INPUT(reshape_13_gm, TensorType({ DT_FLOAT16 })) /* "3nd operand." */
    .INPUT(anchor13_gm, TensorType({ DT_FLOAT16 }))
    .INPUT(anchor26_gm, TensorType({ DT_FLOAT16 }))
    .INPUT(anchor52_gm, TensorType({ DT_FLOAT16 }))
    .INPUT(grid13_gm, TensorType({ DT_FLOAT16 }))
    .INPUT(grid26_gm, TensorType({ DT_FLOAT16 }))
    .INPUT(gird52_gm, TensorType({ DT_FLOAT16 }))
    .INPUT(mask_gm, TensorType({ DT_FLOAT16 }))
    .INPUT(objclass_gm, TensorType({ DT_FLOAT16 }))
    .OUTPUT(output_gm, TensorType({ DT_FLOAT16 })) /* "Result" */
    .OP_END_FACTORY_REG(Yolov3PostProcessor)
}

#endif // GE_OP_TIK_H
