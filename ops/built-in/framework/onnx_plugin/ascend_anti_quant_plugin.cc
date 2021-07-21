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

Status ParseParamsAscendAntiQuant(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  float scale = 0.0;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      scale = attr.f();
      break;
    }
  }

  float offset = 0.0;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "offset" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      offset = attr.f()*(-1);
      break;
    }
  }

  op_dest.SetAttr("scale", scale);
  op_dest.SetAttr("offset", offset);

  int dtype = 0;  // DT_FLOAT
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "dtype" && attr.type() == ge::onnx::AttributeProto::INT) {
      dtype = attr.i();
      break;
    }
  }

  bool sqrt_mode = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "sqrt_mode" && attr.i() != 0) {
      sqrt_mode = true;
      break;
    }
  }

  op_dest.SetAttr("dtype", dtype);
  op_dest.SetAttr("sqrt_mode", sqrt_mode);

  return SUCCESS;
}

// register AscendAntiQuant op info to GE
REGISTER_CUSTOM_OP("AscendAntiQuant")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::AscendAntiQuant",
                   "ai.onnx::9::AscendAntiQuant",
                   "ai.onnx::10::AscendAntiQuant",
                   "ai.onnx::11::AscendAntiQuant",
                   "ai.onnx::12::AscendAntiQuant",
                   "ai.onnx::13::AscendAntiQuant"})
    .ParseParamsFn(ParseParamsAscendAntiQuant)
    .ImplyType(ImplyType::TVM);
}  // namespace domi