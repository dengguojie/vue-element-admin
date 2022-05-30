/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
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
#include "selection_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
Status ParseParamsTopK(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE("TopK", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int axis = -1;
  bool sorted = true;
  bool largest = true;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "sorted" && attr.type() == ge::onnx::AttributeProto::INT) {
      sorted = static_cast<bool>(attr.i());
    } else if (attr.name() == "largest" && attr.type() == ge::onnx::AttributeProto::INT) {
      largest = static_cast<bool>(attr.i());
    } else if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis = attr.i();
    }
  }

  op_dest.SetAttr("name", node->name());
  op_dest.SetAttr("sorted", sorted);
  op_dest.SetAttr("dim", axis);
  op_dest.SetAttr("largest", largest);

  return SUCCESS;
}

Status ParseParamsTopKV9(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE("TopK", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);

  // 1.add dynamic input and out
  int op_input_size = node->input_size();
  int op_output_size = node->output_size();
  opDesc->AddDynamicInputDesc("x", op_input_size);
  opDesc->AddDynamicOutputDesc("y", op_output_size);

  // 2.set original_type
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::9::TopK");
  int axis = -1;
  int64_t data = -1;
  
  // 3.set attr if needed
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "k" && attr.type() == ge::onnx::AttributeProto::INT) {
      data = attr.i();
    }
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis = attr.i();
    }
  }
  if (data == -1) {
    ONNX_PLUGIN_LOGE("TopK", "onnx TopK has no K attr.");
    return PARAM_INVALID;
  }

  std::vector<int64_t> dims = {};
  const ge::Tensor value_tensor = Scalar2Tensor(data, dims, ge::DT_INT32); 
  op_dest.SetAttr("k", value_tensor);
  op_dest.SetAttr("dim", axis);
  op_dest.SetAttr("name", node->name());
  return SUCCESS;
}

static Status ParseOpToGraphTopKV9(const ge::Operator& op, ge::Graph& graph) {
  std::string ori_name;
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get name from op failed.");
    return FAILED;
  }

  auto data0 = op::Data(ori_name + "_data0").set_attr_index(0);
  ge::Tensor k_value;
  if (op.GetAttr("k", k_value) != SUCCESS) {
    ONNX_PLUGIN_LOGE("TopK", "get value from op failed");
    return FAILED;
  }
  auto data1 = op::Const(ori_name + "_data1").set_attr_value(k_value);
  auto topk = op::TopK(ori_name + "_TopK").set_input_x(data0).set_input_k(data1);
  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  output_indexs.emplace_back(topk, vector<std::size_t>{0});
  output_indexs.emplace_back(topk, vector<std::size_t>{1});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::TopK", "ai.onnx::9::TopK"})
    .ParseParamsFn(ParseParamsTopKV9)
    .ParseOpToGraphFn(ParseOpToGraphTopKV9)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("TopK")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::10::TopK", "ai.onnx::11::TopK", "ai.onnx::12::TopK", "ai.onnx::13::TopK",
                   "ai.onnx::14::TopK", "ai.onnx::15::TopK"})
    .ParseParamsFn(ParseParamsTopK)
    .ImplyType(ImplyType::TVM);

}  // namespace domi
