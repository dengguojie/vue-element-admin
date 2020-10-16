/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file roipooling_ops.h
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

IMPLEMT_INFERFUNC(ROIPooling, ROIPoolingInferShape) {

    auto pooled_h = op.get_attr_pooled_h();
    auto pooled_w = op.get_attr_pooled_w();

    auto xShape = op.get_input_desc_x().GetShape().GetDims();
    auto xDtype = op.get_input_desc_x().GetDataType();
    auto roisShape = op.get_input_desc_rois().GetShape().GetDims();
    auto roi_max_num = roisShape[2];

    int64_t inputN, inputC1,poolH, poolW;
    inputN = xShape[0];
    inputC1 = xShape[1];

    poolH = pooled_h;
    poolW = pooled_w;

    vector<int64_t> yShape({roi_max_num*inputN, inputC1, poolH, poolW});

    auto outdesc = op.GetOutputDesc("y");
    outdesc.SetShape(Shape(yShape));
    outdesc.SetDataType(ge::DataType(xDtype));
    (void)op.update_output_desc_y(outdesc);

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ROIPooling, ROIPoolingVerify) {
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ROIPooling, ROIPoolingInferShape);
VERIFY_FUNC_REG(ROIPooling, ROIPoolingVerify);

}
