/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the
License.
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

// #### Set param in attr for transfer
Status ParseParamsArgMaxWithK(const Message* op_src,  ge::Operator&  op_dest)
{
    OP_LOGI("ArgMaxWithK", "Start into the ParseParamsArgMaxWithK!");
    const caffe::LayerParameter* layer = static_cast<const caffe::LayerParameter*>(op_src);

    if (nullptr == layer)
    {
        OP_LOGE("ArgMaxWithK", "Dynamic cast op_src to LayerParameter failed.");
        return FAILED;
    }

    const caffe::ArgMaxParameter& argmax_param = layer->argmax_param();

    //Parse axis
    if(argmax_param.has_axis())
    {
        int axis = (int)argmax_param.axis();
        op_dest.SetAttr("axis", axis);
    }
    else{
        op_dest.SetAttr("axis", 10000);
    }

    //Parse out_max_val
    if(argmax_param.has_out_max_val())
    {
        bool out_max_val = (bool)argmax_param.out_max_val();
        op_dest.SetAttr("out_max_val", out_max_val);
    }
    else{
        op_dest.SetAttr("out_max_val", false);
    }

    //Parse top_k
    if(argmax_param.has_top_k())
    {
        int top_k = (int)argmax_param.top_k();
        op_dest.SetAttr("topk", top_k);
    }
    else{
        op_dest.SetAttr("topk", 1);
    }

    OP_LOGI("ArgMaxWithK", "End of the ParseParamsArgMaxWithK!");

    return SUCCESS;
}

REGISTER_CUSTOM_OP("ArgMaxWithK")
    .FrameworkType(CAFFE)  // type: CAFFE, TENSORFLOW
    .OriginOpType("ArgMax")  // name in caffe module
    .ParseParamsFn(ParseParamsArgMaxWithK)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);

}  // namespace domi
