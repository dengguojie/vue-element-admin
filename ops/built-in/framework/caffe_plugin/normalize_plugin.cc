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
Status ParseParamsNormalize(const Message* op_src,  ge::Operator&  op_dest)
{
    OP_LOGI("Normalize", "Start into the ParseParamsNormalize!");
    const caffe::LayerParameter* layer = static_cast<const caffe::LayerParameter*>(op_src);

    if (nullptr == layer)
    {
        OP_LOGE("Normalize", "Dynamic cast op_src to LayerParameter failed.");
        return FAILED;
    }

    const caffe::NormalizeParameter& normalize_param = layer->norm_param();

    //Parse across_spatial
    if(!normalize_param.has_across_spatial())
    {
        op_dest.SetAttr("across_spatial", true);
    }
    else{
        bool across_spatial = (bool)normalize_param.across_spatial();
        op_dest.SetAttr("across_spatial", across_spatial);
    }

    //Parse channel_shared
    if (!normalize_param.has_channel_shared())
    {
        op_dest.SetAttr("channel_shared", true);
    }
    else{
        bool channel_shared = normalize_param.channel_shared();
        op_dest.SetAttr("channel_shared", channel_shared);
    }

    //Parse eps
    if (!normalize_param.has_eps())
    {
        op_dest.SetAttr("eps", (float)(1e-10));
    }
    else{
        float eps = (float)normalize_param.eps();
        op_dest.SetAttr("eps", eps);
    }

    OP_LOGI("Normalize", "End of the ParseParamsNormalize!");

    return SUCCESS;
}

REGISTER_CUSTOM_OP("Normalize")
    .FrameworkType(CAFFE)  // type: CAFFE, TENSORFLOW
    .OriginOpType("Normalize")  // name in caffe module
    .ParseParamsFn(ParseParamsNormalize)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);

}  // namespace domi
