/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Create: 2020-08-27
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

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
using NodeProto = ge::onnx::NodeProto;
// only support onnx flatten axis be [0,r)
Status ParseParamsFlatten(const Message* op_src, ge::Operator& op_dest) {
  // 1.add dynamic input and out
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (opDesc == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Get OpDesc From operator failed.");
    return FAILED;
  }
  opDesc->AddDynamicInputDesc("args", 1);
  opDesc->AddDynamicOutputDesc("output", 1);

  // 2.set attr if needed
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int axis = 1;
  for (auto attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis = attr.i();
    }
  }
  op_dest.SetAttr("axis", axis);

  return SUCCESS;
}

Status ParseParamsFlattenV11(const Message* op_src, ge::Operator& op_dest) {
  if (ParseParamsFlatten(op_src, op_dest) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Parser params of flatten failed.");
    return FAILED;
  }
  // set original_type
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::11::Flatten");

  return SUCCESS;
}

Status ParseParamsFlattenV12(const Message* op_src, ge::Operator& op_dest) {
  if (ParseParamsFlatten(op_src, op_dest) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Parser params of flatten failed.");
    return FAILED;
  }
  // set original_type
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::12::Flatten");

  return SUCCESS;
}

Status ParseParamsFlattenV13(const Message* op_src, ge::Operator& op_dest) {
  if (ParseParamsFlatten(op_src, op_dest) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Parser params of flatten failed.");
    return FAILED;
  }
  // set original_type
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::13::Flatten");

  return SUCCESS;
}

Status ParseParamsFlattenV9(const Message* op_src, ge::Operator& op_dest) {
  if (ParseParamsFlatten(op_src, op_dest) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Parser params of flatten failed.");
    return FAILED;
  }
  // set original_type
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::9::Flatten");

  return SUCCESS;
}

static Status ParseOpToGraphFlatten(const Operator& op, Graph& graph) {
  int axis = 1;
  if (op.GetAttr("axis", axis) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get axis from op failed");
    return FAILED;
  }
  if (axis < 0) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "negative axis[%d] is not support.", axis);
    return FAILED;
  }

  auto data0 = op::Data("data0").set_attr_index(0);
  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
  if (axis == 0) {
    auto flattenV2 = op::FlattenV2().set_input_x(data0).set_attr_axis(0).set_attr_end_axis(-1);

    std::vector<int64_t> dims = {1};
    ge::Shape shape(dims);
    TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
    int32_t tmpAxis = 0;
    ge::Tensor tensorAxis(tensorDesc, reinterpret_cast<uint8_t*>(&tmpAxis), sizeof(ge::DT_INT32));
    auto data1 = op::Const("data1").set_attr_value(tensorAxis);
    auto expandDims = op::ExpandDims().set_input_x(flattenV2).set_input_axis(data1);

    output_indexs.emplace_back(expandDims, vector<std::size_t>{0});
  } else {
    auto flattenV2_1 = op::FlattenV2().set_input_x(data0).set_attr_axis(0).set_attr_end_axis(axis - 1);
    auto flattenV2_2 = op::FlattenV2().set_input_x(flattenV2_1).set_attr_axis(1).set_attr_end_axis(-1);

    output_indexs.emplace_back(flattenV2_2, vector<std::size_t>{0});
  }

  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

// register Flatten op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::Flatten")
  .ParseParamsFn(ParseParamsFlattenV11)
  .ParseOpToGraphFn(ParseOpToGraphFlatten)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::12::Flatten")
  .ParseParamsFn(ParseParamsFlattenV12)
  .ParseOpToGraphFn(ParseOpToGraphFlatten)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::13::Flatten")
  .ParseParamsFn(ParseParamsFlattenV13)
  .ParseOpToGraphFn(ParseOpToGraphFlatten)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::Flatten",
                 "ai.onnx::9::Flatten",
                 "ai.onnx::10::Flatten"})
  .ParseParamsFn(ParseParamsFlattenV9)
  .ParseOpToGraphFn(ParseOpToGraphFlatten)
  .ImplyType(ImplyType::TVM);
}  // namespace domi