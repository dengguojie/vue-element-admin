/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Create: 2020-08-27
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "onnx_common.h"

using domi::ONNX;

namespace domi {
Status ParseParamsSqueeze(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::vector<int64_t> axes = {};
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        axes.push_back(attr.ints(i));
      }
    }
  }

  op_dest.SetAttr("axis", axes);
  return SUCCESS;
}

// register Add op info to GE
REGISTER_CUSTOM_OP("Squeeze")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Squeeze",
                   "ai.onnx::9::Squeeze",
                   "ai.onnx::10::Squeeze",
                   "ai.onnx::11::Squeeze",
                   "ai.onnx::12::Squeeze"})
    .ParseParamsFn(ParseParamsSqueeze)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
