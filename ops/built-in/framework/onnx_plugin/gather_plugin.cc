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
 * \file gather_plugin.cpp
 * \brief
 */

#include "onnx_common.h"
#include "selection_ops.h"
#include "array_ops.h"

using namespace ge;
namespace domi {

static const int DEFAULT_AXIS = 0;

Status ParseParamsGather(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  op_dest.SetAttr("original_type", "ai.onnx::11::Gather");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", 2);
  op_desc->AddDynamicOutputDesc("y", 1);
  
  int32_t axis_val = DEFAULT_AXIS;

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis_val = attr.i();
    }
  }

  std::vector<int64_t> value_dims = {1};
  ge::Tensor tensor1 = Scalar2Tensor(axis_val, value_dims, ge::DT_INT32);
  op_dest.SetAttr("constant_values", tensor1);
  return SUCCESS;
}

static Status ParseOpToGraphGather(const ge::Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);
  auto data1 = op::Data("data1").set_attr_index(1);

  ge::Tensor const_value;
  if (op.GetAttr("constant_values", const_value) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get const_value from op failed");
    return FAILED;
  }

  auto const_op = op::Const("const_data").set_attr_value(const_value);
  
  auto gather2v_op = op::GatherV2("gatherv2").set_input_x(data0)
                                             .set_input_indices(data1)
                                             .set_input_axis(const_op);

  std::vector<ge::Operator> inputs = {data0, data1, const_op};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(gather2v_op, std::vector<size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Gather",
                   "ai.onnx::9::Gather",
                   "ai.onnx::10::Gather",
                   "ai.onnx::11::Gather",
                   "ai.onnx::12::Gather",
                   "ai.onnx::13::Gather"})
    .ParseParamsFn(ParseParamsGather)
    .ParseOpToGraphFn(ParseOpToGraphGather)
    .ImplyType(ImplyType::TVM);
}  // namespace domi

