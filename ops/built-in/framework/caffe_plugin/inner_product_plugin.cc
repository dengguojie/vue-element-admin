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
Status ParseParamsInnerProduct(const Message* op_src,  ge::Operator&  op_dest)
{
    OP_LOGI("InnerProduct",
            "Start into the ParseParamsInnerProduct!");
    const caffe::LayerParameter* layer = static_cast<const caffe::LayerParameter*>(op_src);

    if (nullptr == layer)
    {
        OP_LOGE("InnerProduct",
            "Dynamic cast op_src to LayerParameter failed");
        return FAILED;
    }
    
    const caffe::InnerProductParameter& inner_product_param = layer->inner_product_param();

    //Parse num_output
    if(!inner_product_param.has_num_output())
    {
        OP_LOGE("InnerProduct",
            "Parse num_output for %s failed.", layer->name().c_str());
        return PARAM_INVALID;
    } else {
        int32_t num_output = (uint32_t)inner_product_param.num_output();
        op_dest.SetAttr("num_output", num_output);
    }
    
    // Parse axis info
    if (!inner_product_param.has_axis())
    {
        op_dest.SetAttr("axis", (int64_t)(1));
    } else {
        int32_t axis = (uint32_t)inner_product_param.axis();
        op_dest.SetAttr("axis", axis);
    }    
    
    //Parse transpose
    if (!inner_product_param.has_transpose())
    {
        op_dest.SetAttr("transpose", false);
    } else {
        bool transpose = inner_product_param.transpose();
        op_dest.SetAttr("transpose", transpose);
    }
    
    op_dest.SetAttr("alpha", (int64_t)(1));
    op_dest.SetAttr("beta", (int64_t)(0));
    
    OP_LOGI("InnerProduct",
            "End of the ParseParamsInnerProduct!");

    return SUCCESS;
}

REGISTER_CUSTOM_OP("FullyConnection")
    .FrameworkType(CAFFE)  // type: CAFFE, TENSORFLOW
    .OriginOpType("InnerProduct")  // name in caffe module
    .ParseParamsFn(ParseParamsInnerProduct)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);

}  // namespace domi
