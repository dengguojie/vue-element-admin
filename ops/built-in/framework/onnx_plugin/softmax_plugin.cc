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
Status ParseParamsSoftmax(const Message *op_src, ge::Operator &op_dest, std::vector<int> &v_axis) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      v_axis.push_back(attr.i());
    }
  }
  
  return SUCCESS;
}

Status ParseParamsSoftmaxV11(const Message *op_src, ge::Operator &op_dest) {
  std::vector<int> v_axis;
  auto ret = ParseParamsSoftmax(op_src, op_dest, v_axis);
  if (ret != SUCCESS) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "acquire attr from NodeProto failed.");
    return FAILED;
  }
  op_dest.SetAttr("axes", v_axis);
  op_dest.SetAttr("need_fusion", 1);
  return SUCCESS;
}

Status ParseParamsSoftmaxV13(const Message *op_src, ge::Operator &op_dest) {
  std::vector<int> v_axis;
  auto ret = ParseParamsSoftmax(op_src, op_dest, v_axis);
  if (ret != SUCCESS) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "acquire attr from NodeProto failed.");
    return FAILED;
  }
  if (v_axis.empty()) {
    // default value in version 13.
    v_axis.push_back(-1);
  }
  op_dest.SetAttr("axes", v_axis);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("SoftmaxV2")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::Softmax",
                 "ai.onnx::9::Softmax",
                 "ai.onnx::10::Softmax",
                 "ai.onnx::11::Softmax",
                 "ai.onnx::12::Softmax"})
  .ParseParamsFn(ParseParamsSoftmaxV11)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("SoftmaxV2")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::13::Softmax")
  .ParseParamsFn(ParseParamsSoftmaxV13)
  .ImplyType(ImplyType::TVM);
}  // namespace domi