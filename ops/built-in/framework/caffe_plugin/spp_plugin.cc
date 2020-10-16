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
namespace domi {
// Caffe ParseParams
Status ParseParamsSPP(const Message* op_origin,
                      ge::Operator& op_dest) {
  OP_LOGI("SPP", "enter into ParseParams SPP ------begin!!\n");
  // trans op_src to op_dest
  const caffe::LayerParameter* layer = \
    dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("SPP", "Dynamic cast op_src to LayerParameter failed.\n");
    return FAILED;
  }

  // get layer
  const caffe::SPPParameter& param = \
    layer->spp_param();
  if (param.has_pyramid_height()) {
    op_dest.SetAttr("pyramid_height", (int64_t)param.pyramid_height());
  }
  if (param.has_pool()) {
    op_dest.SetAttr("pool_method", param.pool());
  }

  OP_LOGI("SPP", "ParseParamsSPP ------end!!\n");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("SPP")
    .FrameworkType(CAFFE)
    .OriginOpType("SPP")
    .ParseParamsFn(ParseParamsSPP)
    .ImplyType(ImplyType::TVM);
}  // namespace domi

