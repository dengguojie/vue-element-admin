/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Plugin for Operator Clip
 * Author: 
 * Create: 2020-06-12
 */
#include "register/register.h"

namespace domi {

Status ParseParamsClip(const ge::Operator &op_src, ge::Operator &op_dst)
{
    float min = 0.0;
    float max = 0.0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("min", min)) {
        op_dst.SetAttr("min", min);
    }
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("max", max)) {
        op_dst.SetAttr("max", max);
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("Clip")
    .FrameworkType(CAFFE)
    .OriginOpType("Clip")
    .ParseParamsByOperatorFn(ParseParamsClip)
    .ImplyType(ImplyType::TVM);
}
