/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "op_log.h"
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"

namespace domi {

/* Exp Attr */
const std::string EXP_ATTR_BASE = "base";
const std::string EXP_ATTR_SCALE = "scale";
const std::string EXP_ATTR_SHIFT = "shift";
const float DEFAULT_BASE_VALUE = -1.0;
const float DEFAULT_SCALE_VALUE = 1.0;
const float DEFAULT_SHIFT_VALUE = 0.0;


Status ParseParamsExp(const Message *op_src, ge::Operator &op_dest)
{
    OP_LOGI("Exp", "--ParseParamsExp  start--");
    const caffe::LayerParameter *layer = static_cast<const caffe::LayerParameter *>(op_src);
    if (layer == nullptr) {
        OP_LOGE("Exp",
            "Dynamic cast op_src to LayerParameter failed");
        return FAILED;
    }
    const caffe::ExpParameter  &exp_param = layer->exp_param();
    if (exp_param.has_base()) {
        op_dest.SetAttr(EXP_ATTR_BASE, (float)exp_param.base());
    } else {
        op_dest.SetAttr(EXP_ATTR_BASE, (float)DEFAULT_BASE_VALUE);
    }
    if (exp_param.has_scale()) {
        op_dest.SetAttr(EXP_ATTR_SCALE, (float)exp_param.scale());
    } else {
        op_dest.SetAttr(EXP_ATTR_SCALE, (float)DEFAULT_SCALE_VALUE);
    }
    if (exp_param.has_shift()) {
        op_dest.SetAttr(EXP_ATTR_SHIFT, (float)exp_param.shift());
    } else {
        op_dest.SetAttr(EXP_ATTR_SHIFT, (float)DEFAULT_SHIFT_VALUE);
    }
    OP_LOGI("Exp", "--ParseParamsExp end--");

    return SUCCESS;
}

// register exp op info to GE
REGISTER_CUSTOM_OP("Exp")
    .FrameworkType(CAFFE)
    .OriginOpType("Exp")
    .ParseParamsFn(ParseParamsExp)
    .ImplyType(ImplyType::TVM);
}  // namespace domi

