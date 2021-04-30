/**
Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
Description: plugin for interp caffe operator
Author:
Create: 2020-6-11
*/

#include "register/register.h"
#include <memory>
#include <string>
#include <vector>

namespace domi {

const char ATTR_HEIGHT[] = "height";
const char ATTR_WIDTH[] = "width";
const char ATTR_ZOOM_FACTOR[] = "zoom_factor";
const char ATTR_SHRINK_FACTOR[] = "shrink_factor";

Status ParseParamsInterp(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int zoom_factor = 0;
    int shrink_factor = 0;
    int nn_height = 1;
    int nn_width = 1;

    if (ge::GRAPH_SUCCESS == op_src.GetAttr(ATTR_SHRINK_FACTOR, shrink_factor) &&
        ge::GRAPH_SUCCESS != op_src.GetAttr(ATTR_ZOOM_FACTOR, zoom_factor)) {
        op_dest.SetAttr(ATTR_SHRINK_FACTOR, shrink_factor);
    } else if (ge::GRAPH_SUCCESS == op_src.GetAttr(ATTR_ZOOM_FACTOR, zoom_factor) &&
        ge::GRAPH_SUCCESS != op_src.GetAttr(ATTR_SHRINK_FACTOR, shrink_factor)) {
        op_dest.SetAttr(ATTR_ZOOM_FACTOR, zoom_factor);
    } else if (
        ge::GRAPH_SUCCESS == op_src.GetAttr(ATTR_HEIGHT, nn_height) &&
        ge::GRAPH_SUCCESS == op_src.GetAttr(ATTR_WIDTH, nn_width)) {
        op_dest.SetAttr(ATTR_HEIGHT, nn_height);
        op_dest.SetAttr(ATTR_WIDTH, nn_width);
    } else if (
        ge::GRAPH_SUCCESS == op_src.GetAttr(ATTR_SHRINK_FACTOR, shrink_factor) &&
        ge::GRAPH_SUCCESS == op_src.GetAttr(ATTR_ZOOM_FACTOR, zoom_factor)) {
        op_dest.SetAttr(ATTR_SHRINK_FACTOR, shrink_factor);
        op_dest.SetAttr(ATTR_ZOOM_FACTOR, zoom_factor);
    } else {
        printf("[ERROR][Plugin] Interp parameters error\n");
        return FAILED;
    }

    return SUCCESS;
}

REGISTER_CUSTOM_OP("Interp")
    .FrameworkType(CAFFE)
    .OriginOpType("Interp")
    .ParseParamsByOperatorFn(ParseParamsInterp)
    .ImplyType(ImplyType::TVM);
}
