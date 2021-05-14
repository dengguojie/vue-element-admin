/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Plugin for operator gru
 * Author: 
 * Create: 2020-06-17
 */

#include "register/register.h"

using namespace ge;

namespace domi { 
    // register GruTik op info to GE
    REGISTER_CUSTOM_OP("GruTik")
        .FrameworkType(TENSORFLOW)
        .OriginOpType("GruTik")
        .ParseParamsFn(AutoMappingFn)
        .ImplyType(ImplyType::TVM);
}
