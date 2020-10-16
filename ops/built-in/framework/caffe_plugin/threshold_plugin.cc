/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this
 * file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http:// www.apache.org/licenses/LICENSE-2.0
 */

#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi
{

Status ParseParamsThreshold(const Message *op_src, ge::Operator &op_dest)
{

  OP_LOGI("Threshold", "enter into ParseParams Threshold ------begin!!");

  if (op_src == nullptr)
  {
    OP_LOGE("Threshold", "op_origin get failed.");
    return FAILED;
  }

  const caffe::LayerParameter *layer = dynamic_cast<const caffe::LayerParameter *>(op_src);
  if (layer == nullptr)
  {
    OP_LOGE("Threshold", "[Threshold_Plugin] cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::ThresholdParameter &param = layer->threshold_param();
  if (param.has_threshold())
  {
    op_dest.SetAttr("threshold", param.threshold());
  }

  OP_LOGI("Threshold", "ParseParams Threshold ------end!!");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("Threshold")
    .FrameworkType(CAFFE)
    .OriginOpType("Threshold")
    .ParseParamsFn(ParseParamsThreshold)
    .ImplyType(ImplyType::TVM);
} // namespace domi
