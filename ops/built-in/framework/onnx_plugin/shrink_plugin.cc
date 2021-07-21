/* Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
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
static const float BIAS_DEFAULT = 0.0;
static const float LAMBD_DEFAULT = 0.5;

Status ParseParamsShrink(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  float bias = BIAS_DEFAULT;
  float lambd = LAMBD_DEFAULT;
  
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "bias" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
        bias = attr.f();
    } 
    if (attr.name() == "lambd" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
        lambd = attr.f();
    }
  }
  op_dest.SetAttr("bias", bias);
  op_dest.SetAttr("lambd", lambd);
  
  return SUCCESS;
}

// register Shrink op info to GE
REGISTER_CUSTOM_OP("Shrink")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::Shrink",
                   "ai.onnx::10::Shrink",
                   "ai.onnx::11::Shrink",
                   "ai.onnx::12::Shrink",
                   "ai.onnx::13::Shrink"})
    .ParseParamsFn(ParseParamsShrink)
    .ImplyType(ImplyType::TVM);
}  // namespace domi