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
#include "array_ops.h"
namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsSoftmaxCrossEntropyLoss(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  for (const auto &attr : node->attribute()) {
    if (attr.name() == "ignore_index" && attr.type() == ge::onnx::AttributeProto::INT) {
      int64_t ignore_index = attr.i();
      op_dest.SetAttr("ignore_index", ignore_index);
    } 
    if (attr.name() == "reduction" && attr.type() == ge::onnx::AttributeProto::STRING) {
      std::string reduction = attr.s();
      op_dest.SetAttr("reduction", reduction);
    }
  }

  return SUCCESS;
}
// register SoftmaxCrossEntropyLoss op info to GE
REGISTER_CUSTOM_OP("SoftmaxCrossEntropyLoss")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::12::SoftmaxCrossEntropyLoss",
                 "ai.onnx::13::SoftmaxCrossEntropyLoss"})
  .ParseParamsFn(ParseParamsSoftmaxCrossEntropyLoss)
  .ImplyType(ImplyType::TVM);
}  // namespace domi