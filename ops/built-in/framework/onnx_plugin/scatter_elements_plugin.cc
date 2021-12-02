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
 * \file scatter_elements_plugin.cc
 * \brief
 */
#include "onnx_common.h"
#include "array_ops.h"

namespace domi {
Status ParseParamsScatterElements(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  
  int axis = 0;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "axis") {
      axis = attr.i();
    }
  }
  op_dest.SetAttr("axis", axis);
  return SUCCESS;
}

// register Add op info to GE
REGISTER_CUSTOM_OP("ScatterElements")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::11::ScatterElements",
                   "ai.onnx::12::ScatterElements",
                   "ai.onnx::13::ScatterElements",
                   "ai.onnx::9::Scatter",
                   "ai.onnx::10::Scatter",
                   "ai.onnx::11::Scatter"})
    .ParseParamsFn(ParseParamsScatterElements)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
