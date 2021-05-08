/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: SRelu算子插件
 * Author: Huawei
 * Create: 2020-6-12
 */

#include <memory>
#include <string>
#include <vector>
#include "register/register.h"

namespace domi {
Status ParseParamsSrelu(const ge::Operator &op_src, ge::Operator &op_dst)
{
    bool channelShared = false;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("channel_shared", channelShared)) {
        op_dst.SetAttr("channel_shared", channelShared);
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("SReLU")
    .FrameworkType(CAFFE)
    .OriginOpType("SReLU")
    .ParseParamsByOperatorFn(ParseParamsSrelu)
    .ImplyType(ImplyType::TVM);
} 
