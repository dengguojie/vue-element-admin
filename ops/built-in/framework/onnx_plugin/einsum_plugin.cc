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
Status ParseParamsEinsum(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  std::string equation_value;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "equation" && attr.type() == ge::onnx::AttributeProto::STRING) {
      equation_value = attr.s();
    }
  }
  if (equation_value.empty()) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Attr of equation must be not null.");
    return FAILED;
  }
  int input_size = node->input_size();
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  opDesc->AddDynamicInputDesc("x", input_size);
  op_dest.SetAttr("equation", equation_value);
  op_dest.SetAttr("N", input_size);
  return SUCCESS;
}
// register Einsum op info to GE
REGISTER_CUSTOM_OP("Einsum")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Einsum", "ai.onnx::9::Einsum", "ai.onnx::10::Einsum", "ai.onnx::11::Einsum",
                   "ai.onnx::12::Einsum", "ai.onnx::13::Einsum"})
    .ParseParamsFn(ParseParamsEinsum)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
