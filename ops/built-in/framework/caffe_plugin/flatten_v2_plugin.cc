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
Status ParseParamsFlattenV2(const Message* op_origin,
                            ge::Operator& op_dest) {
  OP_LOGI("FlattenV2",
            "enter into ParseParams FlattenV2 ------begin!!");
  // trans op_src to op_dest
  const caffe::LayerParameter* layer = \
    dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("FlattenV2",
            "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  // get layer
  const caffe::FlattenParameter& param = \
    layer->flatten_param();
  if (param.has_axis()) {
    op_dest.SetAttr("axis", param.axis());
  }
  if (param.has_end_axis()) {
    op_dest.SetAttr("end_axis", param.end_axis());
  }

  OP_LOGI("FlattenV2",
            "ParseParamsFlattenV2 ------end!!");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("FlattenV2")
    .FrameworkType(CAFFE)
    .OriginOpType("Flatten")
    .ParseParamsFn(ParseParamsFlattenV2)
    .ImplyType(ImplyType::TVM);
}  // namespace domi

