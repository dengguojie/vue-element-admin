/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
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

namespace domi {

Status ParseParamsAscendRequantS16(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  bool dual_output = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "dual_output" && attr.i() != 0) {
      dual_output = true;
      break;
    }
  }

  bool relu_flag = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "relu_flag" && attr.i() != 0) {
      relu_flag = true;
      break;
    }
  }

  op_dest.SetAttr("dual_output", dual_output);
  op_dest.SetAttr("relu_flag", relu_flag);

  return SUCCESS;
}

// register AscendRequantS16 op info to GE
REGISTER_CUSTOM_OP("AscendRequantS16")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::AscendRequantS16",
                   "ai.onnx::9::AscendRequantS16",
                   "ai.onnx::10::AscendRequantS16",
                   "ai.onnx::11::AscendRequantS16",
                   "ai.onnx::12::AscendRequantS16",
                   "ai.onnx::13::AscendRequantS16"})
    .ParseParamsFn(ParseParamsAscendRequantS16)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
