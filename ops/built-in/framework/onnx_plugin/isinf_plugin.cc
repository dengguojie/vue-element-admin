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
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsIsinf(const Message *op_src, ge::Operator &op_dst) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dst.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int detect_negative = 1;
  int detect_positive = 1;
  for (auto attr : node->attribute()) {
    if (attr.name() == "detect_negative" && attr.type() == ge::onnx::AttributeProto::INT) {
      detect_negative = attr.i();
    } else if (attr.name() == "detect_positive" && attr.type() == ge::onnx::AttributeProto::INT) {
      detect_positive = attr.i();
    }
  }
  if (detect_negative != 1 || detect_positive != 1) {
    ONNX_PLUGIN_LOGE(op_dst.GetName().c_str(), "The operator will computes both positive and negative cases!");
    return FAILED;
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("IsInf")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::10::IsInf",
                 "ai.onnx::11::IsInf",
                 "ai.onnx::12::IsInf",
                 "ai.onnx::13::IsInf"})
  .ParseParamsFn(ParseParamsIsinf)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
