/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
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
// Caffe ParseParams
Status ParseParamsReduction(const Message* op_src, ge::Operator& op_dst)
{
    const caffe::LayerParameter* layer =
        dynamic_cast<const caffe::LayerParameter*>(op_src);

    if (nullptr == layer)
    {
        OP_LOGE("Reduction", "Dynamic cast op_src to LayerParameter failed.");
        return FAILED;
    }
    
    const caffe::ReductionParameter& param = layer->reduction_param();
    
    if(param.has_operation())
    {
        op_dst.SetAttr("operation", (int64_t)param.operation());
    }
    
    if(param.has_axis())
    {
        op_dst.SetAttr("axis", (int64_t)param.axis());
    }
    
    if(param.has_coeff())
    {
        op_dst.SetAttr("coeff", param.coeff());
    }

    return SUCCESS;
}


// register Reduction operation
REGISTER_CUSTOM_OP("Reduction")
    .FrameworkType(CAFFE)  // type: CAFFE, TENSORFLOW
    .OriginOpType("Reduction")  // name in caffe module
    .ParseParamsFn(ParseParamsReduction)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);

}  // namespace domi
