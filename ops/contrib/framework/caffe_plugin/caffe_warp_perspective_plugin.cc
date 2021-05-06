/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: warpPerspective_plugin.cpp c3x pasrser cpp file
 * Author: Huawei
 * Create: 2020-6-11
 * Note:
 */

#include <string>
#include "register/register.h"

using namespace std;

namespace domi {
// transform caffe recognized data structure to ge recognized
Status ParseParamsWarpPerspective(const ge::Operator &op_src, ge::Operator &op_dst)
{
    int constant_value = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("constant_value", constant_value)) {
        op_dst.SetAttr("constant_value", constant_value);
    }
    string interpolation = "";
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("interpolation", interpolation)) {
        op_dst.SetAttr("interpolation", interpolation);
    }
    int dst_height = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("dst_height", dst_height)) {
        op_dst.SetAttr("dst_height", dst_height);
    }
    int dst_width = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("dst_width", dst_width)) {
        op_dst.SetAttr("dst_width", dst_width);
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("WarpPerspective")
    .FrameworkType(CAFFE)
    .OriginOpType("WarpPerspective")
    .ParseParamsByOperatorFn(ParseParamsWarpPerspective)
    .ImplyType(ImplyType::TVM);
}
