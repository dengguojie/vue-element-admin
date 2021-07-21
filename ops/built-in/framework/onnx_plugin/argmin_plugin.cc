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

using namespace ge;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsArgMin(const Message *op_src, ge::Operator &op_dest) {
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (opDesc == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Get OpDesc from operator failed.");
    return FAILED;
  }

  opDesc->AddDynamicInputDesc("x", 1);
  opDesc->AddDynamicOutputDesc("y", 1);
  // 2.set original_type
  op_dest.SetAttr("original_type", "ai.onnx::11::ArgMin");

  // 3.set attr if needed
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int axis = 0;
  int keep_dims = 1;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis = attr.i();
      ONNX_PLUGIN_LOGI(op_dest.GetName().c_str(), "set dimension = %d", axis);
    }
    if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
      keep_dims = attr.i();
    }
    if (attr.name() == "select_last_index" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 1) {
      ONNX_PLUGIN_LOGW(op_dest.GetName().c_str(), "select_last_index should be 0, but now it's 1");
    }
  }

  op_dest.SetAttr("dimension", axis);
  op_dest.SetAttr("keep_dims", keep_dims);
  return SUCCESS;
}

static Status ParseOpToGraphArgMin(const Operator& op, Graph& graph) {
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

  auto argMinD = op::ArgMinD().set_input_x(data0).set_attr_dimension(axis).set_attr_dtype(ge::DT_INT64);

  if (keep_dims == 1) {
    std::vector<int64_t> dims = {1};
    ge::Shape shape(dims);
    TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
    ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(&axis), sizeof(ge::DT_INT32));
    auto data1 = op::Const("data1").set_attr_value(tensor);

    auto expandDims = op::ExpandDims().set_input_x(argMinD).set_input_axis(data1);
    outputs.emplace_back(expandDims, vector<std::size_t>{0});
  } else {
    outputs.emplace_back(argMinD, vector<std::size_t>{0});
  }

  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}


// register ArgMin op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::ArgMin",
                   "ai.onnx::9::ArgMin",
                   "ai.onnx::10::ArgMin",
                   "ai.onnx::11::ArgMin",
                   "ai.onnx::12::ArgMin",
                   "ai.onnx::13::ArgMin"})
    .ParseParamsFn(ParseParamsArgMin)
    .ParseOpToGraphFn(ParseOpToGraphArgMin)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
