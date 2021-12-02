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
#include "bitwise_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsBitShift(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (opDesc == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Get OpDesc from operator failed.");
    return FAILED;
  }
  opDesc->AddDynamicInputDesc("x", 2);
  opDesc->AddDynamicOutputDesc("y", 1);
  // 2.set original_type
  op_dest.SetAttr("original_type", "ai.onnx::11::BitShift");

  std::string direction = "";
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "direction" && attr.type() == ge::onnx::AttributeProto::STRING) {
      direction = attr.s();
    } else {
      ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "direction attr is empty");
      return FAILED;
    }
  }

  op_dest.SetAttr("direction", direction);
  return SUCCESS;
}

static Status ParseOpToGraphBitShift(const Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);
  auto data1 = op::Data("data1").set_attr_index(1);
  std::string direction = "RIGHT";
  if (op.GetAttr("direction", direction) != SUCCESS) {
      ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get value from op failed");
      return FAILED;
  }

  std::vector<Operator> inputs{data0, data1};
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  if (direction == "RIGHT") {
    auto bitshift_r = op::RightShift().set_input_x(data0).set_input_y(data1);
    output_indexs.emplace_back(bitshift_r, vector<std::size_t>{0});
  } else if (direction == "LEFT") {
    auto bitshift_l = op::LeftShift().set_input_x(data0).set_input_y(data1);
    output_indexs.emplace_back(bitshift_l, vector<std::size_t>{0});
  } else {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "attr direction is error");
    return FAILED;
  }

  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}


// register BitShift op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::11::BitShift",
                 "ai.onnx::12::BitShift",
                 "ai.onnx::13::BitShift"})
  .ParseParamsFn(ParseParamsBitShift)
  .ParseOpToGraphFn(ParseOpToGraphBitShift)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
