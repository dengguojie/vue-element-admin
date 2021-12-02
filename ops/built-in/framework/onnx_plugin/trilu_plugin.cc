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
#include "matrix_calculation_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
Status ParseParamsTrilu(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int upper = 1;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "upper" && attr.type() == ge::onnx::AttributeProto::INT) {
      upper = attr.i();
    }
  }
  
  if (node->input_size() == 2) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "now only support one input, not support k input");
    return FAILED;
  }
  op_dest.SetAttr("upper", upper);
  op_dest.SetAttr("original_type", "ai.onnx::14::Trilu");
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  opDesc->AddDynamicInputDesc("x", 1);
  opDesc->AddDynamicOutputDesc("y", 1);
  return SUCCESS;
}

static Status ParseOpToGraphTrilu(const ge::Operator& op, ge::Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);
  int upper = 1;
  op.GetAttr("upper", upper);
  ge::Operator trilu;
  if (upper) {
    trilu = op::Triu("triu").set_input_x(data0).set_attr_diagonal(0);
  } else {
    trilu = op::Tril("tril").set_input_x(data0).set_attr_diagonal(0);
  }

  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  output_indexs.emplace_back(trilu, vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::14::Trilu"})
    .ParseParamsFn(ParseParamsTrilu)
    .ParseOpToGraphFn(ParseOpToGraphTrilu)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
