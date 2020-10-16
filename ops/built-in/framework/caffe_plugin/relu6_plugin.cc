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
namespace domi
{
Status ParseParamsRelu6(const Message* op_origin, ge::Operator& op_dest)
{
  const caffe::LayerParameter* layer = \
    dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("Relu6", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::ReLUParameter& param = layer->relu_param();
  if (param.has_negative_slope()) {
    op_dest.SetAttr("negative_slope", param.negative_slope());
  } else {
    op_dest.SetAttr("negative_slope", float(0));
  }

  return SUCCESS;
}

// register ReLU6 op info to GE
REGISTER_CUSTOM_OP("Relu6")
  .FrameworkType(CAFFE)
  .OriginOpType("ReLU6")
  .ParseParamsFn(ParseParamsRelu6)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
