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

using ge::Operator;
using namespace ge;

namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsMean(const Message* op_src, ge::Operator& op_dest) {
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (opDesc == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Get OpDesc from operator failed.");
    return FAILED;
  }
  op_dest.SetAttr("original_type", "ai.onnx::9::Mean");
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed");
    return FAILED;
  }
  int n_num = node->input_size();

  op_dest.SetAttr("N", n_num);
  opDesc->AddDynamicInputDesc("x", n_num);
  opDesc->AddDynamicOutputDesc("y", 1);

  return SUCCESS;
}

Status ParseOpToGraphMean(const ge::Operator& op, Graph& graph) {
  int n_num = 0;
  if (op.GetAttr("N", n_num) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get attribute N failed");
    return FAILED;
  }
  std::vector<ge::Operator> inputs;
  std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
  if (n_num == 0) {
    ONNX_PLUGIN_LOGE("ParseOpToGraphMean", "input size must greater than 1");
    return FAILED;
  } else {
    auto acc = op::AccumulateNV2("AccumulateNV2").create_dynamic_input_x(n_num).set_attr_N(n_num);
    std::string input_name = "";
    for (int i = 0; i < n_num; ++i) {
      input_name = "data_mean_" + to_string(i);
      auto data_op = op::Data(input_name).set_attr_index(i);
      acc.set_dynamic_input_x(i, data_op);
      inputs.push_back(data_op);
    }

    float num = 1.0 / n_num;
    auto div_op = op::Muls().set_input_x(acc).set_attr_value(num);
    output_indexs.emplace_back(div_op, std::vector<size_t>{0});
  }
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::Mean",
                 "ai.onnx::9::Mean",
                 "ai.onnx::10::Mean",
                 "ai.onnx::11::Mean",
                 "ai.onnx::12::Mean",
                 "ai.onnx::13::Mean"})
  .ParseParamsFn(ParseParamsMean)
  .ParseOpToGraphFn(ParseOpToGraphMean)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
