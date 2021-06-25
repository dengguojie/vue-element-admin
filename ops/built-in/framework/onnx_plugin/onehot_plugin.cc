/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file onehot_plugin.cc
 * \brief
 */

#include "onnx_common.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;

Status ParseParamsOnehotCall(const Message* op_src, ge::Operator& op_dest) {
   const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int axis = -1;
  for (auto& attr : node->attribute()) {
    if (attr.name() == "axis") {
      axis = attr.i();
      break;
    }
  }

  op_dest.SetAttr("axis", axis);
  OpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (op_desc == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "op_desc is null");
    return FAILED;
  }
  op_desc->AddDynamicInputDesc("x", 3);
  op_desc->AddDynamicOutputDesc("y", 1);
  op_dest.SetAttr("original_type", "ai.onnx::11::OneHot");
  return SUCCESS;
}

Status ParseOpToGraphOnehot(const ge::Operator& op, Graph& graph) {
  int axis = -1;
  op.GetAttr("axis", axis);

  auto data1 = op::Data("indicate").set_attr_index(0);
  auto data2 = op::Data("depth").set_attr_index(1);
  auto data3 = op::Data("values").set_attr_index(2);
  
  int32_t split_dim = 0;
  std::vector<int64_t> dims;
  ge::Shape shape(dims);
  TensorDesc tensor_desc(shape, ge::FORMAT_ND, ge::DT_INT32);
  ge::Tensor split_dim_tensor(tensor_desc, reinterpret_cast<uint8_t*>(&split_dim), sizeof(ge::DT_INT32));
  auto split_const_op = op::Const("split_dim").set_attr_value(split_dim_tensor);
  //In order to solve the problem that the input is negative insert add and mod, becase tbe onehot not support but onnx support
  auto split_d_op = op::Split("Split").create_dynamic_output_y(2)
                                      .set_input_x(data3)
                                      .set_input_split_dim(split_const_op)
                                      .set_attr_num_split(2);

  auto cast_op = op::Cast("Cast").set_input_x(data1).set_attr_dst_type(3);
  auto cast_op1 = op::Cast("Cast1").set_input_x(data2).set_attr_dst_type(3);
  auto add_op = op::Add("Add").set_input_x1(cast_op, 0).set_input_x2(cast_op1, 0);
  auto mod_op = op::Mod("Mod").set_input_x1(add_op, 0).set_input_x2(cast_op1, 0);

  auto onehot_op = op::OneHot("OneHot").set_input_x(mod_op, 0)
                                       .set_input_depth(cast_op1, 0)
                                       .set_input_on_value(split_d_op, 1)
                                       .set_input_off_value(split_d_op, 0)
                                       .set_attr_axis(axis);
  std::vector<ge::Operator> inputs{data1, data2, data3};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(onehot_op, std::vector<size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::9::OneHot",
                 "ai.onnx::10::OneHot",
                 "ai.onnx::11::OneHot",
                 "ai.onnx::12::OneHot",
                 "ai.onnx::13::OneHot"})
  .ParseParamsFn(ParseParamsOnehotCall)
  .ParseOpToGraphFn(ParseOpToGraphOnehot)
  .ImplyType(ImplyType::TVM);
}  // namespace domi