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
 * \file xor_plugin.cpp
 * \brief
 */
#include "onnx_common.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "bitwise_ops.h"

using namespace ge;
namespace domi {

Status parseParamsXor(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  op_dest.SetAttr("original_type", "ai.onnx::11::Xor");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", 2);
  op_desc->AddDynamicOutputDesc("y", 1);

  return SUCCESS;
}

static Status ParseOpToGraphXor(const ge::Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);
  auto data1 = op::Data("data1").set_attr_index(1);

  auto cast_a = op::Cast("Cast_a").set_input_x(data0).set_attr_dst_type(ge::DT_INT32);
  auto cast_b = op::Cast("Cast_b").set_input_x(data1).set_attr_dst_type(ge::DT_INT32);
  auto bit_wise_xor = op::BitwiseXor("Xor").set_input_x1(cast_a).set_input_x2(cast_b);
  auto cast_out = op::Cast("Cast_out").set_input_x(bit_wise_xor).set_attr_dst_type(ge::DT_BOOL);

  std::vector<ge::Operator> inputs = {data0, data1};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(cast_out, std::vector<size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Xor", "ai.onnx::9::Xor", "ai.onnx::10::Xor", "ai.onnx::11::Xor", "ai.onnx::12::Xor",
                   "ai.onnx::13::Xor"})
    .ParseParamsFn(parseParamsXor)
    .ParseOpToGraphFn(ParseOpToGraphXor)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
