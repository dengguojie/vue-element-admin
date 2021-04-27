/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Plugin for operator rank
 * Author:
 * Create: 2020-06-17
 */
#include "register/register.h"

using namespace ge;

namespace domi
{
    REGISTER_CUSTOM_OP("TikRank")
        .FrameworkType(TENSORFLOW)
        .OriginOpType("Rank")
        .ParseParamsFn(AutoMappingFn)
        .ImplyType(ImplyType::TVM);
}
