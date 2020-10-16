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
namespace domi{
/*
* parse bias parameters
* param[in] op_src  source op description
* param[out] op_dst destination op description
* return SUCCESS:parse success
*        FAILED: parse failed
*/
Status ParseParamsBias(const Message* op_src, ge::Operator& op_dest) {
    OP_LOGI("Bias",
    "------Parse Params for Bias begin------");
    // trans op_src to op_dest
    const caffe::LayerParameter* layer =
    dynamic_cast<const caffe::LayerParameter*>(op_src);
    if (nullptr == layer) {
        OP_LOGE("Bias",
        "Dynamic cast op_src to LayerParameter failed.");
        return FAILED;
    }

    const caffe::BiasParameter& param = layer->bias_param();

    if (param.has_axis()) {
        op_dest.SetAttr("axis", (int)(param.axis()));
    }
    if (param.has_num_axes()) {
        op_dest.SetAttr("num_axes", (int)(param.num_axes()));
    }

    int n = layer->bottom_size();
    if (n > 1) {
        op_dest.SetAttr("bias_from_blob", false);
    }
    else {
        op_dest.SetAttr("bias_from_blob", true);
    }

    OP_LOGI("Bias",
    "------Parse Params for Bias end------");
    return SUCCESS;
}

// test_reduction is the type name of the operator in the OM model.
// It can be specified randomly and cannot be the same as an existing type name. It is case sensitive.
REGISTER_CUSTOM_OP("Bias")
    .FrameworkType(CAFFE)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("Bias")  // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParamsBias)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);
}  // namespace domi
