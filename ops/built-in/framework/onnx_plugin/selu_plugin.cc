/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file selu_plugin.cpp
 * \brief
 */
 #include <cmath>
#include "onnx_common.h"
#include "array_ops.h"

namespace domi {
static const float ALPHA_DEFAULT = 1.67326324235;
static const float GAMMA_DEFAULT = 1.05070098736;
static const float DIFF_ABS = 0.00001;
Status ParseParamsSelu(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  float alpha = ALPHA_DEFAULT;
  float gamma = GAMMA_DEFAULT;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "alpha" && attr.type() == ge::onnx::AttributeProto::FLOAT) alpha = attr.f();
    if (attr.name() == "gamma" && attr.type() == ge::onnx::AttributeProto::FLOAT) gamma = attr.f();
  }
  if(fabs(alpha - ALPHA_DEFAULT) > DIFF_ABS || fabs(gamma - GAMMA_DEFAULT) > DIFF_ABS) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "the attr of alpha or gamma is not default. transform failed.");
    return FAILED;
  }

  return SUCCESS;
}

// register Add op info to GE
REGISTER_CUSTOM_OP("Selu")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Selu",
                   "ai.onnx::9::Selu",
                   "ai.onnx::10::Selu",
                   "ai.onnx::11::Selu",
                   "ai.onnx::12::Selu",
                   "ai.onnx::13::Selu"})
    .ParseParamsFn(ParseParamsSelu)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
