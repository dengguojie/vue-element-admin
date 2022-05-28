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
#include "reduce_ops.h"

using namespace ge;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsReduceMax(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::vector<int> axes = {};
  bool keep_dims = true;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        axes.push_back(attr.ints(i));
      }
    } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
      keep_dims = (attr.i() == 1);
    }
  }
  int num = axes.size();
  std::vector<int64_t> dims = {};
  if (num != 0) {
    dims.push_back(num);
  }
  ge::Tensor tensor = Vec2Tensor(axes, dims, ge::DT_INT32, ge::FORMAT_NCHW);

  op_dest.SetAttr("name", node->name());
  op_dest.SetAttr("axes", tensor);
  op_dest.SetAttr("keep_dims", keep_dims);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (op_desc == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Get OpDesc from operator failed.");
    return FAILED;
  }
  op_desc->AddDynamicInputDesc("x", 2);
  op_desc->AddDynamicOutputDesc("y", 1);
  op_dest.SetAttr("original_type", "ai.onnx::11::ReduceMax");

  return SUCCESS;
}

static Status ParseOpToGraphReduceMax(const ge::Operator& op, Graph& graph) {
  std::string ori_name;
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get name from op failed.");
    return FAILED;
  }

  auto data0 = op::Data(ori_name + "_data0").set_attr_index(0);

  ge::Tensor axes;
  if (op.GetAttr("axes", axes) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get axes from op failed");
    return FAILED;
  }
  auto data1 = op::Const(ori_name + "_data1").set_attr_value(axes);
  auto reducemax = op::ReduceMax(ori_name + "_ReduceMax").set_input_x(data0).set_input_axes(data1);

  bool keep_dims = false;
  if (op.GetAttr("keep_dims", keep_dims) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get keep_dims from op failed");
    return FAILED;
  }
  reducemax.set_attr_keep_dims(keep_dims);

  std::vector<ge::Operator> inputs{data0};
  std::vector<std::pair<ge::Operator, std::vector<size_t> > > outputs;
  outputs.emplace_back(reducemax, std::vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(outputs);

  return SUCCESS;
}

// register ReduceMax op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::ReduceMax",
                   "ai.onnx::9::ReduceMax",
                   "ai.onnx::10::ReduceMax",
                   "ai.onnx::11::ReduceMax",
                   "ai.onnx::12::ReduceMax",
                   "ai.onnx::13::ReduceMax",
                   "ai.onnx::14::ReduceMax",
                   "ai.onnx::15::ReduceMax"})
    .ParseParamsFn(ParseParamsReduceMax)
    .ParseOpToGraphFn(ParseOpToGraphReduceMax)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
