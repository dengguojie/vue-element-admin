/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file threshold_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
Status ParseParamsThreshold(const Message* op_src, ge::Operator& op_dest)
{
  OP_LOGI("Threshold", "enter into ParseParams Threshold ------begin!!");

  if (op_src == nullptr) {
    OP_LOGE("Threshold", "op_origin get failed.");
    return FAILED;
  }

  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);
  if (layer == nullptr) {
    OP_LOGE("Threshold", "[Threshold_Plugin] cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::ThresholdParameter& param = layer->threshold_param();
  if (param.has_threshold()) {
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
}  // namespace domi
