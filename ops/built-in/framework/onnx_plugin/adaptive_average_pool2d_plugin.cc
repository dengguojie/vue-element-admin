/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file adaptive_average_pool2d_plugin.cc
 * \brief
 */
#include "onnx_common.h"

namespace domi {

Status ParseParamsAdaptiveAvgPool2d(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  std::vector<int> v_output_size = {};
  bool set_output_size_flag = false;

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "output_size" && attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.ints_size() == 2) {
        for (int i = 0; i < attr.ints_size(); i++) {
          v_output_size.push_back(attr.ints(i));
        }
      } else {
        ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "length of output_size must be 2.");
      }
      set_output_size_flag = true;
    }
  }

  if (set_output_size_flag) {
    op_dest.SetAttr("output_size", v_output_size);
  } else {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "onnx AdaptiveAvgPool2d op has no output_size attr.");
  }
  return SUCCESS;
}

// register AdaptiveMaxPool2d op info to GE
REGISTER_CUSTOM_OP("AdaptiveAvgPool2d")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::AdaptiveAvgPool2d", "ai.onnx::9::AdaptiveAvgPool2d", "ai.onnx::10::AdaptiveAvgPool2d",
                   "ai.onnx::11::AdaptiveAvgPool2d", "ai.onnx::12::AdaptiveAvgPool2d",
                   "ai.onnx::13::AdaptiveAvgPool2d"})
    .ParseParamsFn(ParseParamsAdaptiveAvgPool2d)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
