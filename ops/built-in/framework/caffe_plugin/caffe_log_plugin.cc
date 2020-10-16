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

    Status ParseParamsLog(const Message* op_src,  ge::Operator&  op_dest)
    {
        OP_LOGI("Log", "--------------ParseParamsLog  start---------------");

        const caffe::LayerParameter* layer = static_cast<const caffe::LayerParameter*>(op_src);
        if(nullptr == layer)
        {
            OP_LOGE("Log", "Dynamic cast op_src to LayerParameter failed.");
            return FAILED;
        }

        const caffe::LogParameter& log_param = layer->log_param();

        float base = log_param.base();
        op_dest.SetAttr("base", (float)base);

        float scale = log_param.scale();
        op_dest.SetAttr("scale", (float)scale);

        float shift = log_param.shift();
        op_dest.SetAttr("shift", (float)shift);

        return SUCCESS;
    }

REGISTER_CUSTOM_OP("Log")
    .FrameworkType(CAFFE)  // type: CAFFE, TENSORFLOW
    .OriginOpType("Log")  // name in caffe module
    .ParseParamsFn(ParseParamsLog)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}  // namespace domi
