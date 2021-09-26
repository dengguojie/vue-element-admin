/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file lp_normalization_plugin.cc
 * \brief
 */
#include "onnx_common.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
Status parse_params_lp_normalization(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  // 1.add dynamic input and out
  op_desc->AddDynamicInputDesc("x", 1);
  op_desc->AddDynamicOutputDesc("y", 1);
  // 2.set original_type
  ge::AttrUtils::SetStr(op_desc, "original_type", "ai.onnx::1::LpNormalization");
  // 3.set attrs
  int axis = -1;
  int p = 2;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis = static_cast<float>(attr.i());
    } else if (attr.name() == "p" && attr.type() == ge::onnx::AttributeProto::INT) {
      p = static_cast<int>(attr.i());
    }
  }
  op_dest.SetAttr("p", p);
  op_dest.SetAttr("axis", axis);
  return SUCCESS;
}

Status parse_op_to_graph_lp_normalization(const Operator& op, Graph& graph) {
  auto data0_op = op::Data("data0").set_attr_index(0);
  auto identity_op = op::Identity().set_input_x(data0_op);
  auto lp_norm_op = op::LpNorm().set_input_x(identity_op);
  auto div_op = op::Div().set_input_x1(identity_op).set_input_x2(lp_norm_op);

  int p_num = 2;
  int axis = -1;
  if (op.GetAttr("p", p_num) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get attr p from op failed");
    return FAILED;
  }
  if (op.GetAttr("axis", axis) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get attr axis from op failed");
    return FAILED;
  }
  lp_norm_op.set_attr_p(p_num);
  lp_norm_op.set_attr_axes({axis});
  lp_norm_op.set_attr_keepdim(true);
  lp_norm_op.set_attr_epsilon(0);

  std::vector<Operator> inputs{data0_op};
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  output_indexs.emplace_back(div_op, vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

// register LpNormalization op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::1::LpNormalization",
                 "ai.onnx::8::LpNormalization",
                 "ai.onnx::9::LpNormalization",
                 "ai.onnx::10::LpNormalization",
                 "ai.onnx::11::LpNormalization",
                 "ai.onnx::12::LpNormalization",
                 "ai.onnx::13::LpNormalization"})
  .ParseParamsFn(parse_params_lp_normalization)
  .ParseOpToGraphFn(parse_op_to_graph_lp_normalization)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
