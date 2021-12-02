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
Status ParseParamsThresholdedRelu(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  float alpha = 1.0;
  for (auto attr : node->attribute()) {
    if (attr.name() == "alpha") {
      alpha = attr.f();
    }
  }
  op_dest.SetAttr("alpha", alpha);
  op_dest.SetAttr("original_type", "ai.onnx::11::ThresholdedRelu");
  OpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", 1);
  op_desc->AddDynamicOutputDesc("y", 1);

  return SUCCESS;
}

Status ParseOpToThresholdedRelu(const ge::Operator &op, Graph &graph) {
  float alpha = 1.0f;
  op.GetAttr("alpha", alpha);
  auto data = op::Data("data1").set_attr_index(0);
  auto identity_op = op::Identity("identity").set_input_x(data);
  auto threshold_op = op::Threshold("threshold").set_input_x(identity_op).set_attr_threshold(alpha);
  auto mul_op = op::Mul("mul").set_input_x1(identity_op).set_input_x2(threshold_op);
  std::vector<ge::Operator> inputs{data};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(mul_op, std::vector<size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::10::ThresholdedRelu",
                 "ai.onnx::11::ThresholdedRelu",
                 "ai.onnx::12::ThresholdedRelu",
                 "ai.onnx::13::ThresholdedRelu"})
  .ParseParamsFn(ParseParamsThresholdedRelu)
  .ParseOpToGraphFn(ParseOpToThresholdedRelu)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
