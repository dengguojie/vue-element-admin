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
 * \file compress_plugin.cpp
 * \brief
 */
#include "onnx_common.h"
#include "array_ops.h"
#include "transformation_ops.h"
#include "selection_ops.h"

using namespace ge;
namespace domi {

Status parseParamsCompress(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  op_dest.SetAttr("original_type", "ai.onnx::11::Compress");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", 3);
  op_desc->AddDynamicOutputDesc("y", 1);
  int32_t axis_val = 0;
  int32_t need_flatten = 1;

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis_val = attr.i();
      need_flatten = 0;
    }
  }

  std::vector<int64_t> value_dims = {1};
  ge::Tensor tensor1 = Scalar2Tensor(axis_val, value_dims, DT_INT32);

  op_dest.SetAttr("need_flatten", need_flatten);
  op_dest.SetAttr("axis_value", tensor1);
  return SUCCESS;
}

static Status ParseOpToGraphCompress(const ge::Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);
  auto data1 = op::Data("data1").set_attr_index(1);

  int need_flatten = 0;
  if (op.GetAttr("need_flatten", need_flatten) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get the switch of whether inserting flatten from op failed");
    return FAILED;
  }

  ge::Tensor const_value;
  if (op.GetAttr("axis_value", const_value) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get const_value from op failed");
    return FAILED;
  }

  auto const_op = op::Const("axis_data").set_attr_value(const_value);

  ge::Operator compress;
  auto where = op::Where("compress_condition").set_input_x(data1);
  if (need_flatten) {
    auto flatten = op::Flatten("compress_flatten").set_input_x(data0).set_attr_axis(0);
    auto squeeze = op::Squeeze("squeeze").set_input_x(flatten).set_attr_axis(0);
    compress = op::GatherV2("compress").set_input_x(squeeze).set_input_indices(where).set_input_axis(const_op);
  } else {
    compress = op::GatherV2("compress").set_input_x(data0).set_input_indices(where).set_input_axis(const_op);
  }

  std::vector<ge::Operator> inputs = {data0, data1, const_op};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(compress, std::vector<size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::Compress", "ai.onnx::10::Compress", "ai.onnx::11::Compress", "ai.onnx::12::Compress",
                   "ai.onnx::13::Compress"})
    .ParseParamsFn(parseParamsCompress)
    .ParseOpToGraphFn(ParseOpToGraphCompress)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
