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
#include "array_ops.h"
#include "elewise_calculation_ops.h"

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
      OP_LOGD(op_dest.GetName().c_str(), "set dimension = %d", axis);
    }
    if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
      keep_dims = attr.i();
    }
    if (attr.name() == "select_last_index" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 1) {
      ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "select_last_index should be 0, but now it's 1");
      return FAILED;
    }
  }
  std::vector<int64_t> value_dims = {};
  ge::Tensor tensor = Scalar2Tensor(axis, value_dims, ge::DT_INT32);
  op_dest.SetAttr("name", node->name());
  op_dest.SetAttr("dimension", tensor);
  op_dest.SetAttr("keep_dims", keep_dims);
  return SUCCESS;
}

static Status ParseOpToGraphArgMin(const Operator& op, Graph& graph) {
  std::string ori_name;
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get name from op failed.");
    return FAILED;
  }

  int keep_dims = 1;
  if (op.GetAttr("keep_dims", keep_dims) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get keep_dims from op failed");
    return FAILED;
  }
  ge::Tensor axis;
  if (op.GetAttr("dimension", axis) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get dimension from op failed");
    return FAILED;
  }

  auto data0 = op::Data(ori_name + "data0").set_attr_index(0);
  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t>>> outputs;
  
  auto const_op = op::Const(ori_name + "const_data").set_attr_value(axis);
  auto argMin = op::ArgMin(ori_name + "ArgMin").set_input_x(data0)
                                               .set_input_dimension(const_op)
                                               .set_attr_dtype(ge::DT_INT32);

  if (keep_dims == 1) {
    std::vector<int64_t> dims = {1};
    ge::Tensor tensor1 = Scalar2Tensor(keep_dims, dims, ge::DT_INT32);
    auto data1 = op::Const(ori_name + "data1").set_attr_value(tensor1);

    auto expandDims = op::ExpandDims(ori_name + "ExpandDims").set_input_x(argMin).set_input_axis(data1);
    outputs.emplace_back(expandDims, vector<std::size_t>{0});
  } else {
    outputs.emplace_back(argMin, vector<std::size_t>{0});
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
                   "ai.onnx::13::ArgMin",
                   "ai.onnx::14::ArgMin",
                   "ai.onnx::15::ArgMin"})
    .ParseParamsFn(ParseParamsArgMin)
    .ParseOpToGraphFn(ParseOpToGraphArgMin)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
