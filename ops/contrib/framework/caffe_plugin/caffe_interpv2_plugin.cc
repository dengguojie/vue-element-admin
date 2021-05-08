/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Plugin for operator interpv2
 * Author:
 * Create: 2020-06-17
 */

#include "register/register.h"

using namespace ge;

namespace domi {
    // transform caffe recognized data structure to ge recognized
    Status CaffeInterpv2ParseParams(const ge::Operator &op_src, ge::Operator &op_dest)
    {
        return SUCCESS;
    }

    REGISTER_CUSTOM_OP("Interpv2")
        .FrameworkType(CAFFE)
        .OriginOpType("Interpv2")
        .ParseParamsByOperatorFn(CaffeInterpv2ParseParams)
        .ImplyType(ImplyType::TVM);
}