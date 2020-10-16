/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"


using namespace ge;
namespace domi
{
// Caffe ParseParams
Status ParseParamsShuffleChannel(const Message* op_origin, ge::Operator& op_dest)
{
    // trans op_src to op_dest
    const caffe::LayerParameter* layer =
        dynamic_cast<const caffe::LayerParameter*>(op_origin);

    if (nullptr == layer) {
        OP_LOGE("ShuffleChannel",
            "Dynamic cast op_src to LayerParameter failed.");
        return FAILED;
    }

    const caffe::ShuffleChannelParameter& param = layer->shuffle_channel_param();
     if(param.has_group()) {
        op_dest.SetAttr("group", (int)param.group());
    }

    return SUCCESS;
}

REGISTER_CUSTOM_OP("ShuffleChannel")
    .FrameworkType(CAFFE)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("ShuffleChannel")  // // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParamsShuffleChannel)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);
}  // namespace domi
