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
 * \file caffe_ascend_quant_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Parse the parameters from caffe model, and pass them to the inner model.
Status ParseParamsAscendQuant(const Message* op_origin, ge::Operator& op_dest) {
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  // Ckeck operator parameter's validity
  if (layer == nullptr) {
    OP_LOGE(op_dest.GetName().c_str(), "convert src op failed.");
    return FAILED;
  }

  // get layer
  const caffe::QuantParameter& param = layer->quant_param();

  if (param.has_scale()) {
    op_dest.SetAttr("scale", param.scale());
  } else {
    op_dest.SetAttr("scale", static_cast<float>(1));
  }

  if (param.has_offset()) {
    op_dest.SetAttr("offset", static_cast<float>(*((signed char*)param.offset().c_str())));
  } else {
    op_dest.SetAttr("offset", static_cast<float>(0));
  }

  string mode = "Round";
  op_dest.SetAttr("sqrt_mode", false);
  op_dest.SetAttr("round_mode", mode);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("AscendQuant")
    .FrameworkType(CAFFE)
    .OriginOpType("Quant")
    .ParseParamsFn(ParseParamsAscendQuant)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
