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

/*
  The input does not need to explicitly be a 2D vector; rather, it will be coerced into one. 
  For an arbitrary n-dimensional tensor input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] 
  and k is the axis provided, then input will be coerced into a 2-dimensional tensor with 
  dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. only in version 12,11,10,9,8
   softmax     ->           data 
                            /  \
                           /    \
                        flatten  shape
                          |        /
                      softmaxv2   /
                          |      /
                          |     /
                         reshape
*/

#include "onnx_common.h"
using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;
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
  op_dest.SetAttr("original_type", "ai.onnx::11::Softmax");
  OpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", 1);
  op_desc->AddDynamicOutputDesc("y", 1);
  return SUCCESS;
}

Status ParseParamsSoftmaxV11(const Message *op_src, ge::Operator &op_dest) {
  std::vector<int> v_axis;
  auto ret = ParseParamsSoftmax(op_src, op_dest, v_axis);
  if (ret != SUCCESS) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "acquire attr from NodeProto failed.");
    return FAILED;
  }
  if (v_axis.empty()) {
    // default value in version 8 9 10 11 12.
    v_axis.push_back(1);
  }
  op_dest.SetAttr("axes", v_axis);
  return SUCCESS;
}

Status ParseOpToGraphSoftmax(const ge::Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);
  auto identity = op::Identity().set_input_x(data0);
  std::vector<int> axis;
  if (op.GetAttr("axes", axis) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "GetAttr axes fail");
    return FAILED;
  }
  if (axis.empty()) {
    axis.push_back(1);
  }
  auto flatten = op::Flatten().set_input_x(identity).set_attr_axis(axis[0]);
  auto shape = op::Shape().set_input_x(identity);
  auto softmax = op::SoftmaxV2().set_input_x(flatten).set_attr_axes({1});
  auto reshape = op::Reshape().set_input_x(softmax).set_input_shape(shape);
  std::vector<ge::Operator> inputs = {data0};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(reshape, std::vector<size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
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

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::Softmax",
                 "ai.onnx::9::Softmax",
                 "ai.onnx::10::Softmax",
                 "ai.onnx::11::Softmax",
                 "ai.onnx::12::Softmax"})
  .ParseParamsFn(ParseParamsSoftmaxV11)
  .ParseOpToGraphFn(ParseOpToGraphSoftmax)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("SoftmaxV2")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::13::Softmax")
  .ParseParamsFn(ParseParamsSoftmaxV13)
  .ImplyType(ImplyType::TVM);
}  // namespace domi