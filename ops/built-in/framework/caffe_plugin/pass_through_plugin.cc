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
Status ParseParamsPassThrough(const Message* op_origin,
                       ge::Operator& op_dest) {
  OP_LOGI("PassThrough", "enter into ParseParams PassThrough ------begin!!");
  // trans op_src to op_dest
  const caffe::LayerParameter* layer = \
    dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("PassThrough", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }
  // get layer
  const caffe::ReorgParameter& param = \
    layer->reorg_param();
  if (param.has_stride()) {
    op_dest.SetAttr("stride", (int32_t)param.stride());
  }
  if (param.has_reverse()) {
      op_dest.SetAttr("reverse", param.reverse());
  }

  OP_LOGI("PassThrough", "ParseParamsPassThrough ------end!!");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("PassThrough")
    .FrameworkType(CAFFE)
    .OriginOpType("Reorg")
    .ParseParamsFn(ParseParamsPassThrough)
    .ImplyType(ImplyType::TVM);
}  // namespace domi


