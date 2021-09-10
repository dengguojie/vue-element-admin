/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2018. All rights reserved.
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
 * \file relu6_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
Status ParseParamsRelu6(const Message* op_origin, ge::Operator& op_dest) {
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("Relu6", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::ReLUParameter& param = layer->relu_param();
  if (param.has_negative_slope()) {
    op_dest.SetAttr("negative_slope", param.negative_slope());
  } else {
    op_dest.SetAttr("negative_slope", static_cast<float>(0));
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
