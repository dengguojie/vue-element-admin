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

using namespace ge;

namespace domi
{
// Caffe ParseParams
Status ParseParamsReverse(const Message* op_src,  ge::Operator&  op_dest)
{
    const caffe::LayerParameter* layer = static_cast<const caffe::LayerParameter*>(op_src);
    if (nullptr == layer)
    {
        return FAILED;
    }

    const caffe::ReverseParameter& reverse_param = layer->reverse_param();

    if (reverse_param.axis_size() == 0) {
        return FAILED;
    }

    vector<int> v_axis;
    for (int i = 0; i < reverse_param.axis_size(); ++i) {
        v_axis.push_back(reverse_param.axis(i));
    }
    op_dest.SetAttr("axis", v_axis);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("ReverseV2D")
    .FrameworkType(CAFFE)  // type: CAFFE, TENSORFLOW
    .OriginOpType("Reverse")  // name in caffe module
    .ParseParamsFn(ParseParamsReverse)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}  // namespace domi
