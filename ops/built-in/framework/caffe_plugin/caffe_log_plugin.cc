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
 * \file caffe_log_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
Status ParseParamsLog(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("Log", "--------------ParseParamsLog  start---------------");

  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);
  if (layer == nullptr) {
    OP_LOGE("Log", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::LogParameter& log_param = layer->log_param();

  float base = log_param.base();
  op_dest.SetAttr("base", static_cast<float>(base));

  float scale = log_param.scale();
  op_dest.SetAttr("scale", static_cast<float>(scale));

  float shift = log_param.shift();
  op_dest.SetAttr("shift", static_cast<float>(shift));

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Log")
    .FrameworkType(CAFFE)           // type: CAFFE, TENSORFLOW
    .OriginOpType("Log")            // name in caffe module
    .ParseParamsFn(ParseParamsLog)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}  // namespace domi
