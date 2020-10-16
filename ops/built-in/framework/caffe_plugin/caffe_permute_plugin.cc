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

/* Permute Attr */
const std::string EXP_ATTR_ORDER = "order";

Status ParseParamsPermute(const Message *op_src, ge::Operator &op_dest)
{
    OP_LOGI("Permute", "ParseParamsPermute start");
    const caffe::LayerParameter *layer = static_cast<const caffe::LayerParameter *>(op_src);
    if (layer == nullptr) {
        OP_LOGE("Permute", "Static cast op_src to LayerParameter failed");
        return FAILED;
    }
    vector<int64_t> orders;
    const caffe::PermuteParameter  &permute_param = layer->permute_param();
    if (0 >= permute_param.order_size()) {
        OP_LOGI("Permute", "Permute layer orders size is less 0, set Defeaut value!");
        orders.push_back((int64_t)0);
    } else {
        // new orders
        for (int i = 0; i < permute_param.order_size(); ++i) {
            uint32_t order = permute_param.order(i);
            if (std::find(orders.begin(), orders.end(), order) != orders.end()) {
                OP_LOGE("Permute", "there are duplicate orders");
                return FAILED;
            }
        orders.push_back((int64_t)order);
        }
    }
    op_dest.SetAttr(EXP_ATTR_ORDER, orders);
    OP_LOGI("Permute", "--ParseParamsPermute  end--");
    return SUCCESS;
}

// register Permute op info to GE
REGISTER_CUSTOM_OP("Permute")
    .FrameworkType(CAFFE)
    .OriginOpType("Permute")
    .ParseParamsFn(ParseParamsPermute);
}  // namespace domi

