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
 * @file psroipooling_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <cmath>
#include <vector>
#include <string>
#include "util/util.h"
#include "op_log.h"
#include "inc/nn_detect_ops.h"

namespace ge {

IMPLEMT_INFERFUNC(PSROIPooling, PSROIPoolingInferShape) {
    auto output_dim = op.get_attr_output_dim();
    auto group_size = op.get_attr_group_size();
    // The value of group_size must be less than 128
    if (group_size >= 128) {
        OP_LOGE(op.GetName().c_str(),
                "The value of group_size not support, is %ld", group_size);
        return GRAPH_FAILED;
    }

    auto x_shape = op.get_input_desc_x().GetShape();
    auto x_dtype = op.get_input_desc_x().GetDataType();
    int64_t pool_h = group_size;
    int64_t pool_w = group_size;

    int64_t c_output_dim = x_shape.GetDim(1) / (group_size * group_size);
    if (c_output_dim != output_dim) {
        OP_LOGE(op.GetName().c_str(),
                "The c of input fm is invalid, is %ld, %ld", x_shape.GetDim(1), output_dim);
        return GRAPH_FAILED;
    }

    auto rois_shape = op.get_input_desc_rois().GetShape();
    int64_t rois_num = rois_shape.GetDim(0) * rois_shape.GetDim(2);

    vector<int64_t> y_shape({rois_num, output_dim, pool_h, pool_w});

    auto out_desc = op.GetOutputDesc("y");
    out_desc.SetShape(Shape(y_shape));
    out_desc.SetDataType(ge::DataType(x_dtype));
    (void)op.update_output_desc_y(out_desc);

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(PSROIPooling, PSROIPoolingVerify) {
    // input x only support NCHW format
    auto x_shape = op.get_input_desc_x().GetShape();
    if (x_shape.GetDims().size() != 4) {
        OP_LOGE(op.GetName().c_str(), "input x shape must be 4d," \
               "input x shape size is %d", x_shape.GetDims().size());
        return GRAPH_FAILED;
    }

    Format x_format = op.get_input_desc_x().GetFormat();
    if (x_format != FORMAT_NCHW) {
        OP_LOGE(op.GetName().c_str(), "input x format must be NCHW");
        return GRAPH_FAILED;
    }

    // rois shape is (batch, 5, rois_num), shape size is 3
    auto rois_shape = op.get_input_desc_rois().GetShape();
    if (rois_shape.GetDims().size() < 3) {
        OP_LOGE(op.GetName().c_str(), "input rois shape must be equal 3," \
               "input rois shape size is %d", rois_shape.GetDims().size());
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(PSROIPooling, PSROIPoolingInferShape);
VERIFY_FUNC_REG(PSROIPooling, PSROIPoolingVerify);

}
