/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Plugin for operator CTCGreedyDecoder
 * Author: Huawei
 * Create: 2020-06-17
 */

#include "register/register.h"
using namespace ge;
namespace domi {
// register ctc_greedy_decoder op info to GE
REGISTER_CUSTOM_OP("ctc_greedy_decoder")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ctc_greedy_decoder")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
