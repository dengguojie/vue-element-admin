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

#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {

Status ParseParamsAbs(const Message *op_src, ge::Operator &op_dest)
{
    const caffe::LayerParameter *layer = static_cast<const caffe::LayerParameter *>(op_src);
    if (layer == nullptr) {
        OP_LOGE("AbsVal", "Static cast op_src to LayerParameter failed");
        return FAILED;
    }
    return SUCCESS;
}

// register AbsVal op info to GE
REGISTER_CUSTOM_OP("Abs")
    .FrameworkType(CAFFE)
    .OriginOpType("AbsVal")
    .ParseParamsFn(ParseParamsAbs);
}  // namespace domi