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
 * \file caffe_leaky_relu_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
Status ParseParamsLeakyRelu(const Message* op_origin, ge::Operator& op_dest) {
  OP_LOGI("LeakyRelu", "enter into ParseParamsLeakyRelu ------begin!!");
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (layer == nullptr) {
    OP_LOGE("LeakyRelu", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }
  // get layer
  const caffe::ReLUParameter& param = layer->relu_param();
  if (param.has_negative_slope()) {
    op_dest.SetAttr("negative_slope", param.negative_slope());
  } else {
    op_dest.SetAttr("negative_slope", static_cast<float>(0));
  }
  return SUCCESS;
}

REGISTER_CUSTOM_OP("LeakyRelu")
    .FrameworkType(CAFFE)                          // type: CAFFE, TENSORFLOW
    .OriginOpType({"ReLU", "RReLU", "LeakyReLU"})  // name in caffe module
    .ParseParamsFn(ParseParamsLeakyRelu)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}  // namespace domi
