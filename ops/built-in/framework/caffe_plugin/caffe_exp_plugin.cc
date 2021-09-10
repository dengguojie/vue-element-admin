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
 * \file caffe_exp_plugin.cpp
 * \brief
 */
#include "op_log.h"
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"

namespace domi {

Status ParseParamsExp(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("Exp", "--ParseParamsExp  start--");
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);
  if (layer == nullptr) {
    OP_LOGE("Exp", "Dynamic cast op_src to LayerParameter failed");
    return FAILED;
  }

  const caffe::ExpParameter& exp_param = layer->exp_param();

  const std::string EXP_ATTR_BASE = "base";
  const float DEFAULT_BASE_VALUE = -1.0;
  if (exp_param.has_base()) {
    op_dest.SetAttr(EXP_ATTR_BASE, static_cast<float>(exp_param.base()));
  } else {
    op_dest.SetAttr(EXP_ATTR_BASE, static_cast<float>(DEFAULT_BASE_VALUE));
  }

  const std::string EXP_ATTR_SCALE = "scale";
  const float DEFAULT_SCALE_VALUE = 1.0;
  if (exp_param.has_scale()) {
    op_dest.SetAttr(EXP_ATTR_SCALE, static_cast<float>(exp_param.scale()));
  } else {
    op_dest.SetAttr(EXP_ATTR_SCALE, static_cast<float>(DEFAULT_SCALE_VALUE));
  }

  const std::string EXP_ATTR_SHIFT = "shift";
  const float DEFAULT_SHIFT_VALUE = 0.0;
  if (exp_param.has_shift()) {
    op_dest.SetAttr(EXP_ATTR_SHIFT, static_cast<float>(exp_param.shift()));
  } else {
    op_dest.SetAttr(EXP_ATTR_SHIFT, static_cast<float>(DEFAULT_SHIFT_VALUE));
  }

  OP_LOGI("Exp", "--ParseParamsExp end--");

  return SUCCESS;
}

// register exp op info to GE
REGISTER_CUSTOM_OP("Exp")
    .FrameworkType(CAFFE)
    .OriginOpType("Exp")
    .ParseParamsFn(ParseParamsExp)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
