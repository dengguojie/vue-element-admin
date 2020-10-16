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
#include "./inc/nn_detect_ops.h"
#include <cmath>
#include <vector>
#include <string>
#include <string.h>
#include "util/util.h"
#include "op_log.h"
#include "graph/debug/ge_log.h"

namespace ge {
IMPLEMT_INFERFUNC(SSDDetectionOutput, SSDDetectionOutputInferShape) {
    auto keep_top_k = op.get_attr_keep_top_k();
    if (keep_top_k == -1) {
        keep_top_k = 1024;
    }

    auto loc_shape = op.get_input_desc_bbox_delta().GetShape().GetDims();
    if (loc_shape.empty()) {
        GE_LOGE("get mbox loc failed.");
        return GRAPH_FAILED;
    }
    auto batch = loc_shape[0];
    auto Boxtype = op.get_input_desc_bbox_delta().GetDataType();

    vector<int64_t> actualNumShape({batch, 8});
    auto outdesc0 = op.GetOutputDesc("out_boxnum");
    outdesc0.SetShape(Shape(actualNumShape));
    outdesc0.SetDataType(ge::DT_INT32);
    (void)op.update_output_desc_out_boxnum(outdesc0);

    vector<int64_t> boxShape({batch, keep_top_k, 8});
    auto outdesc = op.GetOutputDesc("y");
    outdesc.SetShape(Shape(boxShape));
    outdesc.SetDataType(ge::DataType(Boxtype));
    (void)op.update_output_desc_y(outdesc);

    return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(SSDDetectionOutput, SSDDetectionOutputVerify) {

    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(SSDDetectionOutput, SSDDetectionOutputInferShape);
VERIFY_FUNC_REG(SSDDetectionOutput, SSDDetectionOutputVerify);
}
