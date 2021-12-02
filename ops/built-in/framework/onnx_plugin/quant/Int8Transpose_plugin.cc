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
#include "../onnx_common.h"
#include "array_ops.h"
#include "transformation_ops.h"

using namespace ge;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsInt8Transpose(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE("Int8Transpose", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (opDesc == nullptr) {
    ONNX_PLUGIN_LOGE("Int8Transpose", "Get OpDesc from operator failed.");
    return FAILED;
  }
  opDesc->AddDynamicInputDesc("x", 2);
  opDesc->AddDynamicOutputDesc("y", 1);
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::11::Int8Transpose");

  std::vector<int64_t> axes = {};
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        axes.push_back(attr.ints(i));
      }
    }
  }

  int num = axes.size();
  std::vector<int64_t> dims = {num};
  ge::Tensor tensor = Vec2Tensor(axes, dims, ge::DT_INT64);
  op_dest.SetAttr("axes", tensor);
  return SUCCESS;
}

static Status ParseOpToGraphInt8Transpose(const ge::Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);

  ge::Tensor perm;
  if (op.GetAttr("axes", perm) != SUCCESS) {
    ONNX_PLUGIN_LOGE("Int8Transpose", "get perm from op failed");
    return FAILED;
  }

  auto data1 = op::Const("data1").set_attr_value(perm);
  auto int8transpose = op::Transpose().set_input_x(data0).set_input_perm(data1);

  std::vector<ge::Operator> inputs{data0, data1};
  std::vector<std::pair<ge::Operator, std::vector<size_t> > > outputs;
  outputs.emplace_back(int8transpose, std::vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

// register int8transpose op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Int8Transpose",
                   "ai.onnx::9::Int8Transpose",
                   "ai.onnx::10::Int8Transpose",
                   "ai.onnx::11::Int8Transpose",
                   "ai.onnx::12::Int8Transpose",
                   "ai.onnx::13::Int8Transpose"})
    .ParseParamsFn(ParseParamsInt8Transpose)
    .ParseOpToGraphFn(ParseOpToGraphInt8Transpose)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
