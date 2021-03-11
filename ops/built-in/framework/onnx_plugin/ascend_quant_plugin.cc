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
#include <string>
#include <vector>

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {

Status ParseParamsAscendQuant(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("AscendQuant", "Dynamic cast op_src to NodeProto failed.");
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
      offset = attr.f();
      break;
    }
  }

  op_dest.SetAttr("scale", scale);
  op_dest.SetAttr("offset", offset);

  std::string round_mode = "Round";
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "round_mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      round_mode = attr.s();
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

  op_dest.SetAttr("sqrt_mode", sqrt_mode);
  op_dest.SetAttr("round_mode", round_mode);

  return SUCCESS;
}

// register AscendQuant op info to GE
REGISTER_CUSTOM_OP("AscendQuant")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::AscendQuant",
                   "ai.onnx::10::AscendQuant",
                   "ai.onnx::11::AscendQuant",
                   "ai.onnx::12::AscendQuant"})
    .ParseParamsFn(ParseParamsAscendQuant)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
