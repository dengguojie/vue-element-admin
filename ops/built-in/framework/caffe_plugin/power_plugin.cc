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
 * \file power_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParamsPower(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("Power", "------------ParseParamsPower  start-------------");

  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);
  if (nullptr == layer) {
    OP_LOGE("Power", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::PowerParameter& power_param = layer->power_param();

  if (!power_param.has_power()) {
    op_dest.SetAttr("power", static_cast<float>(1.0));
  } else {
    float power = power_param.power();
    op_dest.SetAttr("power", static_cast<float>(power));
  }

  if (!power_param.has_scale()) {
    op_dest.SetAttr("scale", static_cast<float>(1.0));
  } else {
    float scale = power_param.scale();
    op_dest.SetAttr("scale", static_cast<float>(scale));
  }

  if (!power_param.has_shift()) {
    op_dest.SetAttr("shift", static_cast<float>(0.0));
  } else {
    float shift = power_param.shift();
    op_dest.SetAttr("shift", static_cast<float>(shift));
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Power")
    .FrameworkType(CAFFE)             // type: CAFFE, TENSORFLOW
    .OriginOpType("Power")            // name in caffe module
    .ParseParamsFn(ParseParamsPower)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}  // namespace domi
