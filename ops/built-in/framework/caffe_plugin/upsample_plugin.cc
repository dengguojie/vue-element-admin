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
Status ParseParams_Upsample(const Message* op_origin, ge::Operator& op_dest)
{
    // trans op_src to op_dest
    const caffe::LayerParameter* layer =
        dynamic_cast<const caffe::LayerParameter*>(op_origin);

    if (nullptr == layer)
    {
        OP_LOGE("Upsample",
            "Dynamic cast op_src to LayerParameter failed.");
        return FAILED;
    }

    const caffe::UpsampleParameter& param = layer->upsample_param();
    if (param.has_scale() && !param.has_stride() && !param.has_stride_h() && !param.has_stride_h()) {
        OP_LOGW("Upsample",
            "scale means each pixel's scale factor, stride means upsample's shape magnification");
    }
    if(param.has_scale())
    {
        op_dest.SetAttr("scale", float(param.scale()));
    }
    if(param.has_stride())
    {
        op_dest.SetAttr("stride_h", int(param.stride()));
        op_dest.SetAttr("stride_w", int(param.stride()));
    }else{
        op_dest.SetAttr("stride_h", int(param.stride_h()));
        op_dest.SetAttr("stride_w", int(param.stride_w()));
    }
    return SUCCESS;
}
// test_reduction is the type name of the operator in the OM model. 
// It can be specified randomly and cannot be the same as an existing type name. It is case sensitive. 
REGISTER_CUSTOM_OP("Upsample") 
    .FrameworkType(CAFFE)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("Upsample")  // // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParams_Upsample)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);
}  // namespace domi
