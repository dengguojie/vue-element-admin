/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file hard_sigmoid_plugin.cc
 * \brief
 */

#include "onnx_common.h"

namespace domi {

Status parse_params_hard_sigmoid(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  float alpha = 0.2;
  float beta = 0.5;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "alpha" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      alpha = static_cast<float>(attr.f());
    } else if (attr.name() == "beta" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      beta = static_cast<float>(attr.f());
    }
  }
  op_dest.SetAttr("alpha", alpha);
  op_dest.SetAttr("beta", beta);
  return SUCCESS;
}

// register HardSigmoid op info to GE
REGISTER_CUSTOM_OP("HardSigmoid")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::1::HardSigmoid",
                 "ai.onnx::6::HardSigmoid",
                 "ai.onnx::8::HardSigmoid",
                 "ai.onnx::9::HardSigmoid",
                 "ai.onnx::10::HardSigmoid",
                 "ai.onnx::11::HardSigmoid",
                 "ai.onnx::12::HardSigmoid",
                 "ai.onnx::13::HardSigmoid"})
  .ParseParamsFn(parse_params_hard_sigmoid)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
