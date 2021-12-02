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
Status ParseParamsReverseSequence(const Message *op_src,
                                  ge::Operator &op_dest) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int batch_axis = 1;
  int time_axis = 0;
  // set batch_dim's default value to b,and set seq_dim's default value to 0
  op_dest.SetAttr("batch_dim", batch_axis);
  op_dest.SetAttr("seq_dim", time_axis);
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "batch_axis" &&
        attr.type() == ge::onnx::AttributeProto::INT) {
      batch_axis = attr.i();
      if (batch_axis == 1) {
        op_dest.SetAttr("batch_dim", 1);
      } else {
        op_dest.SetAttr("batch_dim", 0);
      }
    }
    if (attr.name() == "time_axis" &&
        attr.type() == ge::onnx::AttributeProto::INT) {
      time_axis = attr.i();
      op_dest.SetAttr("seq_dim", time_axis);
    }
  }
  return SUCCESS;
}
// register ReverseSequence op info to GE
REGISTER_CUSTOM_OP("ReverseSequence")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::10::ReverseSequence",
                 "ai.onnx::11::ReverseSequence",
                 "ai.onnx::12::ReverseSequence",
                 "ai.onnx::13::ReverseSequence"})
  .ParseParamsFn(ParseParamsReverseSequence)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
