/*
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in complian
ce with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file bitwise_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/nn_detect_ops.h"
#include <cmath>
#include <vector>
#include <string>
#include <string.h>
#include "util/util.h"
#include "op_log.h"
namespace ge {
//----------------FSRDetectionOutput-------------------
IMPLEMT_INFERFUNC(FSRDetectionOutput, FSRDetectionOutputInferShape) {
    auto batch_rois = op.get_attr_batch_rois();
    batch_rois = op.get_input_desc_rois().GetShape().GetDims().at(0);
    auto post_nms_topn = op.get_input_desc_rois().GetShape().GetDims().at(2);
    if (post_nms_topn >= 1024) {
        post_nms_topn = 1024;
    }
    auto num_classes = op.get_attr_num_classes();

    auto priorShape = op.get_input_desc_bbox_delta().GetShape().GetDims();
    auto priorDtype = op.get_input_desc_bbox_delta().GetDataType();
    vector<int64_t> actualNumShape({batch_rois, num_classes, 8});
    auto outdesc0 = op.GetOutputDesc("actual_bbox_num");
    outdesc0.SetShape(Shape(actualNumShape));
    outdesc0.SetDataType(ge::DataType(3));
    (void)op.update_output_desc_actual_bbox_num(outdesc0);
    vector<int64_t> boxShape({batch_rois, num_classes, post_nms_topn, 8});
    auto outdesc = op.GetOutputDesc("box");
    outdesc.SetShape(Shape(boxShape));
    outdesc.SetDataType(ge::DataType(priorDtype));
    (void)op.update_output_desc_box(outdesc);
    return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(FSRDetectionOutput, FSRDetectionOutputVerify) {
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(FSRDetectionOutput, FSRDetectionOutputInferShape);
VERIFY_FUNC_REG(FSRDetectionOutput, FSRDetectionOutputVerify);
}
