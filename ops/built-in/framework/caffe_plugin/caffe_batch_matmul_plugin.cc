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

Status ParseParamBatchMatMul(const Message *op_src, ge::Operator &op_dest)
{
    // set the default adj_x1 and adj_x2 value for BatchMatMul,

    const caffe::LayerParameter* layer = \
        dynamic_cast<const caffe::LayerParameter*>(op_src);
    // Ckeck operator parameter's validity
    if (nullptr == layer) {
        OP_LOGE(op_dest.GetName().c_str(), "convert src op failed.");
        return FAILED;
    }
    // get layer
    const caffe::BatchMatMulParameter& param = layer->batch_matmul_param();
    if (param.has_adj_x1()) {
        op_dest.SetAttr("adj_x1", (bool)param.adj_x1());
    } else {
        op_dest.SetAttr("adj_x1", false);
    }

    if (param.has_adj_x2()) {
        op_dest.SetAttr("adj_x2", (bool)param.adj_x2());
    } else {
        op_dest.SetAttr("adj_x2", false);
    }

  return SUCCESS;
}

// register BatchMatMul op info to GE
REGISTER_CUSTOM_OP("BatchMatMul")
  .FrameworkType(CAFFE)
  .OriginOpType("BatchedMatMul")
  .ParseParamsFn(ParseParamBatchMatMul)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
