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

using namespace ge;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsArgMax(const Message* op_src, ge::Operator& op_dest) {
  // 1.add dynamic input and out
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (opDesc == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Get OpDesc from operator failed.");
    return FAILED;
  }
  opDesc->AddDynamicInputDesc("x", 1);
  opDesc->AddDynamicOutputDesc("y", 1);
  // 2.set original_type
  op_dest.SetAttr("original_type", "ai.onnx::11::ArgMax");
  // 3.set attr if needed
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed");
    return FAILED;
  }

  int axis = 0;
  int keep_dims = 1;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis = attr.i();
    }
    if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
      keep_dims = attr.i();
    }
    if (attr.name() == "select_last_index" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 1) {
      OP_LOGW(op_dest.GetName().c_str(), "Only support select_last_index=0, but 1 is obtained now");
    }
  }
  op_dest.SetAttr("dimension", axis);
  op_dest.SetAttr("keep_dims", keep_dims);
  return SUCCESS;
}

static Status ParseOpToGraphArgMax(const Operator& op, Graph& graph) {
  int keep_dims = 1;
  if (op.GetAttr("keep_dims", keep_dims) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get keep_dims from op failed");
    return FAILED;
  }
  int axis = 0;
  if (op.GetAttr("dimension", axis) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get dimension from op failed");
    return FAILED;
  }

  auto data0 = op::Data("data0").set_attr_index(0);
  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t>>> outputs;

  auto argMaxD = op::ArgMaxD().set_input_x(data0).set_attr_dimension(axis).set_attr_dtype(DT_INT64);

  if (keep_dims == 1) {
    std::vector<int64_t> dims = {1};
    ge::Shape shape(dims);
    TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
    ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(&axis), sizeof(ge::DT_INT32));
    auto data1 = op::Const("data1").set_attr_value(tensor);

    auto expandDims = op::ExpandDims().set_input_x(argMaxD).set_input_axis(data1);
    outputs.emplace_back(expandDims, vector<std::size_t>{0});
  } else {
    outputs.emplace_back(argMaxD, vector<std::size_t>{0});
  }

  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

// register ArgMax op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::ArgMax",
                   "ai.onnx::9::ArgMax",
                   "ai.onnx::10::ArgMax",
                   "ai.onnx::11::ArgMax",
                   "ai.onnx::12::ArgMax",
                   "ai.onnx::13::ArgMax"})
    .ParseParamsFn(ParseParamsArgMax)
    .ParseOpToGraphFn(ParseOpToGraphArgMax)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
