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
 * \file reduce_log_sum_plugin.cc
 * \brief
 */
#include "onnx_common.h"
#include "array_ops.h"
#include "reduce_ops.h"
#include "elewise_calculation_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
Status ParseParamsReduceLogSum(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  int op_input_size = node->input_size();
  int op_output_size = node->output_size();
  opDesc->AddDynamicInputDesc("x", op_input_size);
  opDesc->AddDynamicOutputDesc("y", op_output_size);

  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::11::ReduceLogSum");

  std::vector<int> v_axes = {};
  bool keep_dims = true;

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        v_axes.push_back(attr.ints(i));
      }
    } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
      keep_dims = (attr.i() == 1);
    }
  }

  int64_t len = v_axes.size();
  std::vector<int64_t> dims = {};
  if (len != 0) {
    dims.push_back(len);
  }
  ge::Tensor tensor1 = Vec2Tensor(v_axes, dims, DT_INT32, ge::FORMAT_ND);
  op_dest.SetAttr("axes", tensor1);
  op_dest.SetAttr("keep_dims", keep_dims);

  return SUCCESS;
}

static Status ParseOpToGraphReduceLogSum(const Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);

  bool flag = false;
  if (op.GetAttr("keep_dims", flag) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get value of keep_dims from op failed.");
    return FAILED;
  }
  ge::Tensor const_value;
  if (op.GetAttr("axes", const_value) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get axes from op failed");
    return FAILED;
  }
  
  auto const_op = op::Const("const_data").set_attr_value(const_value);
  auto reducesum = op::ReduceSum().set_input_x(data0)
                                  .set_input_axes(const_op)
				  .set_attr_keep_dims(flag);
  auto log = op::Log().set_input_x(reducesum);

  std::vector<Operator> inputs{data0, const_op};
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  output_indexs.emplace_back(log, vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

// register ReduceLogSum op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::ReduceLogSum",
                   "ai.onnx::9::ReduceLogSum",
                   "ai.onnx::10::ReduceLogSum",
                   "ai.onnx::11::ReduceLogSum",
                   "ai.onnx::12::ReduceLogSum",
                   "ai.onnx::13::ReduceLogSum"})
    .ParseParamsFn(ParseParamsReduceLogSum)
    .ParseOpToGraphFn(ParseOpToGraphReduceLogSum)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
