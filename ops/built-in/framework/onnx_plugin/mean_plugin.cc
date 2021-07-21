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
  int N_num = node->input_size();

  op_dest.SetAttr("N", N_num);
  opDesc->AddDynamicInputDesc("x", N_num);
  opDesc->AddDynamicOutputDesc("y", 1);

  return SUCCESS;
}

Status ParseOpToGraphMean(const ge::Operator& op, Graph& graph) {
  int N_num = 0;
  if (op.GetAttr("N", N_num) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get attribute N failed");
    return FAILED;
  }
  std::vector<ge::Operator> inputs;
  std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
  if (N_num == 0) {
    ONNX_PLUGIN_LOGE("ParseOpToGraphMean", "input size must greater than 1");
    return FAILED;
  } else {
    auto ACC = op::AccumulateNV2("AccumulateNV2").create_dynamic_input_x(N_num).set_attr_N(N_num);
    std::string input_name = "";
    for (int i = 0; i < N_num; ++i) {
      input_name = "data_mean_" + to_string(i);
      auto data_op = op::Data(input_name).set_attr_index(i);
      ACC.set_dynamic_input_x(i, data_op);
      inputs.push_back(data_op);
    }
    ge::TensorDesc tensorDesc2;
    float N_f = N_num;
    vector<float> N_list = {N_f};
    vector<int64_t> dims2 = {1};
    ge::Shape shape(dims2);
    tensorDesc2.SetShape(shape);
    tensorDesc2.SetDataType(DT_FLOAT);
    ge::Tensor Num_Tensor(tensorDesc2, reinterpret_cast<uint8_t*>(N_list.data()), sizeof(float));
    auto data2 = op::Const("data2").set_attr_value(Num_Tensor);
    auto div_op = op::Div().set_input_x1(ACC).set_input_x2(data2);
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
