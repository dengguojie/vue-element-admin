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

using namespace ge;

namespace domi {

    Status ParseParamsBNLL(const Message* op_origin,  ge::Operator&  op_dest)
    {
        // trans op_src to op_dest
        if (op_origin == nullptr){
            OP_LOGE("op_origin get failed.\n");
            return FAILED;
        }
        const caffe::LayerParameter* layer = \
        dynamic_cast<const caffe::LayerParameter*>(op_origin);

        if (layer == nullptr) {
            OP_LOGE("Dynamic cast op_src to LayerParameter failed.\n");
            return FAILED;
        }

        return SUCCESS;
    }

    // register BNLL op info to GE
    REGISTER_CUSTOM_OP("BNLL")
    .FrameworkType(CAFFE)
    .OriginOpType("BNLL")
    .ParseParamsFn(ParseParamsBNLL)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
