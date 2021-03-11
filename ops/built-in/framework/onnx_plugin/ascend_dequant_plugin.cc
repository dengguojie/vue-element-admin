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

Status ParseParamsAscendDequant(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("AscendDequant", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

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

  bool relu_flag = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "relu_flag" && attr.i() != 0) {
      relu_flag = true;
      break;
    }
  }

  op_dest.SetAttr("dtype", dtype);
  op_dest.SetAttr("sqrt_mode", sqrt_mode);
  op_dest.SetAttr("relu_flag", relu_flag);

  return SUCCESS;
}

// register AscendDequant op info to GE
REGISTER_CUSTOM_OP("AscendDequant")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::AscendDequant",
                   "ai.onnx::10::AscendDequant",
                   "ai.onnx::11::AscendDequant",
                   "ai.onnx::12::AscendDequant"})
    .ParseParamsFn(ParseParamsAscendDequant)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
