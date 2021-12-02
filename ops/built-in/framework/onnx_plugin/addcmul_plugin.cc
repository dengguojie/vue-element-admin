/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file add_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

#include "array_ops.h"
#include "elewise_calculation_ops.h"
namespace domi {

Status ParseParamsAddcmul(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  float value = 0.0;
  auto attrs = node->attribute();
  auto it = std::find_if(attrs.begin(), attrs.end(),
            [](const ge::onnx::AttributeProto &attr) -> bool{return attr.name() == "value";});
  if (it != attrs.end()){
    value = it->f();
  } else {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "get attr value failed.");
    return FAILED;
  }
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("args", 3);
  op_desc->AddDynamicOutputDesc("output", 1);
  ge::AttrUtils::SetStr(op_desc, "original_type", "torch.onnx.symbolic_opset11.addcmul");
  op_dest.SetAttr("value", value);
  return SUCCESS;
}

static Status ParseOpToGraph(const ge::Operator &op, ge::Graph &graph) {
  auto data_0 = ge::op::Data().set_attr_index(0);
  auto data_1 = ge::op::Data().set_attr_index(1);
  auto data_2 = ge::op::Data().set_attr_index(2);
  float value = 0.0;
  if (op.GetAttr("value", value) != ge::GRAPH_SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get attr value failed.");
    return FAILED;
  }
  auto mul = ge::op::Mul("Mul");
  mul.set_input_x1(data_1);
  mul.set_input_x2(data_2);
  auto muls = ge::op::Muls("Muls");
  muls.set_input_x(mul);
  muls.set_attr_value(value);
  auto add = ge::op::Add("Add");
  add.set_input_x1(data_0);
  add.set_input_x2(muls);

  std::vector<ge::Operator> inputs{data_0, data_1, data_2};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(add, vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

// register Addcmul op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType("torch.onnx.symbolic_opset11.addcmul")
    .ParseParamsFn(ParseParamsAddcmul)
    .ParseOpToGraphFn(ParseOpToGraph)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
