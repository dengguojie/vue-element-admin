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
 * \file concat_plugin.cpp
 * \brief
 */
#include "onnx_common.h"
#include "split_combination_ops.h"
#include "array_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;
static const int DEFAULT_CONCAT_DIM = 0;

Status OpConcatUpdateInfo(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (nullptr == node) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int concat_dim = 0;
  bool set_axis_flag = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      concat_dim = attr.i();
      set_axis_flag = true;
      break;
    }
  }

  if (!set_axis_flag) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "onnx Concat op has no axis attr.");
    concat_dim = DEFAULT_CONCAT_DIM;
    return PARAM_INVALID;
  }
  op_dest.SetAttr("concat_dim", concat_dim);

  int n = node->input_size();
  op_dest.SetAttr("N", n);
  OpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", n);
  op_desc->AddDynamicOutputDesc("y", 1);
  return SUCCESS;
}

Status ParseParamsConcatCall(const Message* op_src, ge::Operator& op_dest) {
  if (OpConcatUpdateInfo(op_src, op_dest) != SUCCESS) {
    return FAILED;
  }
  op_dest.SetAttr("original_type", "ai.onnx::11::Concat");
  return SUCCESS;
}

Status ParseOpToGraphConcat(const ge::Operator& op, Graph& graph) {
  int input_size = 0;
  op.GetAttr("N", input_size);
  std::vector<ge::Operator> inputs;
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;

  if (input_size == 0) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "input_size must ge 1");
    return FAILED;
  } else if (input_size == 1) {
    auto data_op = op::Data("data").set_attr_index(0);
    auto identity_op = op::Identity("identity").set_input_x(data_op);
    inputs.push_back(data_op);
    output_indexs.emplace_back(identity_op, std::vector<size_t>{0});
  } else {
    int concat_dim = DEFAULT_CONCAT_DIM;
    op.GetAttr("concat_dim", concat_dim);

    auto concat_op = op::ConcatD("ConcatD")
                         .create_dynamic_input_x(input_size)
                         .set_attr_concat_dim(concat_dim)
                         .set_attr_N(input_size);
                         
    std::string input_name = "";
    for (int i = 0; i < input_size; ++i) {
      input_name = "data_concat_" + to_string(i);
      auto data_op = op::Data(input_name).set_attr_index(i);
      concat_op.set_dynamic_input_x(i, data_op);
      inputs.push_back(data_op);
    }
    output_indexs.emplace_back(concat_op, std::vector<size_t>{0});
  }
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Concat",
                   "ai.onnx::9::Concat",
                   "ai.onnx::10::Concat",
                   "ai.onnx::11::Concat",
                   "ai.onnx::12::Concat",
                   "ai.onnx::13::Concat"})
    .ParseParamsFn(ParseParamsConcatCall)
    .ParseOpToGraphFn(ParseOpToGraphConcat)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
