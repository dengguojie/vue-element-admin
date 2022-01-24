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
 * \file unsqueeze_plugin.cpp
 * \brief
 */
#include "onnx_common.h"
#include "array_ops.h"

using domi::ONNX;

namespace domi {

Status ParseParamsUnsqueeze(const Message *op_src, ge::Operator &op_dest) {
  const ge::onnx::NodeProto *node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::vector<int64_t> axes = {};
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        axes.push_back(attr.ints(i));
      }
    }
  }
  if (axes.size() == 0) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "axes is a required argument, please check it!");
    return PARAM_INVALID;
  }
  op_dest.SetAttr("axes", axes);
  return SUCCESS;
}

Status ParseParamsUnsqueezeV3(const Message *op_src, ge::Operator &op_dest) {
  const ge::onnx::NodeProto *node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  return SUCCESS;
}

// register Add op info to GE
REGISTER_CUSTOM_OP("Unsqueeze")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Unsqueeze",
                   "ai.onnx::9::Unsqueeze",
                   "ai.onnx::10::Unsqueeze",
                   "ai.onnx::11::Unsqueeze",
                   "ai.onnx::12::Unsqueeze"})
    .ParseParamsFn(ParseParamsUnsqueeze)
    .ImplyType(ImplyType::GELOCAL);

REGISTER_CUSTOM_OP("UnsqueezeV3")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::13::Unsqueeze"})
    .ParseParamsFn(ParseParamsUnsqueezeV3)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi