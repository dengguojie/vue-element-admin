/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
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
Status ParseParamsPower(const Message* op_src,  ge::Operator&  op_dest)
{
    OP_LOGI("Power", "------------ParseParamsPower  start-------------");

    const caffe::LayerParameter* layer = static_cast<const caffe::LayerParameter*>(op_src);
    if(nullptr == layer)
    {
        OP_LOGE("Power", "Dynamic cast op_src to LayerParameter failed.");
        return FAILED;
    }

    const caffe::PowerParameter& power_param = layer->power_param();

    if(!power_param.has_power())
    {
        op_dest.SetAttr("power", (float)1.0);
    }
    else {
        float power = power_param.power();
        op_dest.SetAttr("power", (float)power);
    }

    if(!power_param.has_scale())
    {
        op_dest.SetAttr("scale", (float)1.0);
    }
    else {
        float scale = power_param.scale();
        op_dest.SetAttr("scale", (float)scale);
    }

    if(!power_param.has_shift())
    {
        op_dest.SetAttr("shift", (float)0.0);
    }
    else {
        float shift = power_param.shift();
        op_dest.SetAttr("shift", (float)shift);
    }

    return SUCCESS;
}

REGISTER_CUSTOM_OP("Power")
    .FrameworkType(CAFFE)  // type: CAFFE, TENSORFLOW
    .OriginOpType("Power")  // name in caffe module
    .ParseParamsFn(ParseParamsPower)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}  // namespace domi
