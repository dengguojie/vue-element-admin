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
using ge::Operator;
namespace domi {

Status ParseParamsTranspose(const Message *op_src, ge::Operator &op_dest) {
  const ge::onnx::NodeProto *node = dynamic_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  // update attribute perm if given
  std::vector<int32_t> perm = {};
  for (auto attr : node->attribute()) {
    if (attr.name() == "perm") {
      for (auto axis : attr.ints()) {
        perm.push_back(axis);
      }
    }
  }

  op_dest.SetAttr("perm", perm);

  // add dynamic input and out
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (opDesc == nullptr){
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Failed to get op_desc from operator.");
    return FAILED;
  }
  opDesc->AddDynamicInputDesc("args", node->input_size());
  opDesc->AddDynamicOutputDesc("output", 1);
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::11::Transpose");
  return SUCCESS;
}


Status ParseOpToGraphTranspose(const ge::Operator& op, Graph & graph){
  std::vector<int32_t> perm;
  if (op.GetAttr("perm", perm) != SUCCESS){
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "Failed to get perm from operator");
    return FAILED;
  }
  // TODO: there is default action in onnx
  if (perm.empty()){
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "must input the attr of perm");
    return FAILED;
  }

  auto data0 = op::Data("data0").set_attr_index(0);
  std::vector<Operator> inputs {data0};
  std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;

  int32_t dim = perm.size();
  auto trans_perm = Vec2Tensor<int32_t>(perm, {dim}, ge::DT_INT32);
  auto const_perm = op::Const().set_attr_value(trans_perm);
  auto transpose = op::Transpose().set_input_x(data0).set_input_perm(const_perm);

  output_indexs.emplace_back(transpose, vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}
// register Transpose op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::Transpose",
                 "ai.onnx::9::Transpose",
                 "ai.onnx::10::Transpose",
                 "ai.onnx::11::Transpose",
                 "ai.onnx::12::Transpose",
                 "ai.onnx::13::Transpose"})
  .ParseParamsFn(ParseParamsTranspose)
  .ParseOpToGraphFn(ParseOpToGraphTranspose)
  .ImplyType(ImplyType::TVM);
}  // namespace domi