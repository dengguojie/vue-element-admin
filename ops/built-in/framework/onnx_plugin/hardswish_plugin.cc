/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file hard_swish_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

using namespace ge;
using ge::Operator;

namespace domi {
static const float DEFAULT_ALPHA = 0.16666666;
static const float DEFAULT_BETA = 0.5;
Status ParseParamsHardSwish(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  opDesc->AddDynamicInputDesc("x", 1);
  opDesc->AddDynamicOutputDesc("y", 1);
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::14::HardSwish");
  return SUCCESS;
}

static Status ParseOpToGraphHardSwish(const Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);
  auto input_x = op::Identity().set_input_x(data0);

  auto ret_hs = op::HardSigmoid().set_input_input_x(input_x)
                                 .set_attr_alpha(DEFAULT_ALPHA)
                                 .set_attr_beta(DEFAULT_BETA);

  auto output = op::Mul().set_input_x1(input_x).set_input_x2(ret_hs);

  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(output, vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

// register HardSwish op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::14::HardSwish"})
  .ParseParamsFn(ParseParamsHardSwish)
  .ParseOpToGraphFn(ParseOpToGraphHardSwish)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
