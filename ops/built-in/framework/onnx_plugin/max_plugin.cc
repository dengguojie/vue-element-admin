/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this
 * file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http:// www.apache.org/licenses/LICENSE-2.0
 */
#include "onnx_common.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;

Status ParseParamsMaxCall(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (nullptr == node) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int n = node->input_size();
  op_dest.SetAttr("N", n);
  OpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", n);
  op_desc->AddDynamicOutputDesc("y", 1);
  op_dest.SetAttr("original_type", "ai.onnx::11::Max");
  return SUCCESS;
}


Status ParseOpToGraphMax(const ge::Operator& op, Graph& graph) {
  int input_size = 0;
  op.GetAttr("N", input_size);
  std::vector<ge::Operator> inputs;
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  if (input_size == 0) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "input_size must >= 1");
    return FAILED;
  } else if (input_size == 1) {
    auto data_op = op::Data("data").set_attr_index(0);
    auto identity_op = op::Identity("identity").set_input_x(data_op);
    inputs.push_back(data_op);
    output_indexs.emplace_back(identity_op, std::vector<size_t>{0});
  } else {
    for (int i = 0; i < input_size; ++i) {
      auto data_op = op::Data().set_attr_index(i);
      inputs.push_back(data_op);
    }
    ge::Operator& prefix_op = inputs[0];
    for (int i = 1; i < input_size; ++i) {
      auto cur_op = op::Maximum().set_input_x1(prefix_op).set_input_x2(inputs[i]);
      prefix_op = cur_op;
    }
    output_indexs.emplace_back(prefix_op, std::vector<size_t>{0});
  }
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Max",
                   "ai.onnx::9::Max",
                   "ai.onnx::10::Max",
                   "ai.onnx::11::Max",
                   "ai.onnx::12::Max",
                   "ai.onnx::13::Max"})
    .ParseParamsFn(ParseParamsMaxCall)
    .ParseOpToGraphFn(ParseOpToGraphMax)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
