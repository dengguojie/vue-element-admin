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
 * \file reduce_sum_plugin.cpp
 * \brief
 */
#include "onnx_common.h"
#include "array_ops.h"
#include "reduce_ops.h"

using namespace ge;
using ge::Operator;

namespace domi {
Status parse_params_reduce_sum(const Message* op_src, ge::Operator& op_dest)
{
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
      ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
      return FAILED;
  }

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);

  opDesc->AddDynamicInputDesc("x", 1);
  opDesc->AddDynamicOutputDesc("output", 1);

  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::11::ReduceSum");

  std::vector<int> v_axes = {};
  bool keep_dims_attr = true;

  for (const auto& attr : node->attribute()) {
      if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
          for (int i = 0; i<attr.ints_size(); i++){
              v_axes.push_back(attr.ints(i));
          }
      } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
          keep_dims_attr = (attr.i() == 1);
      }
  }

  int num = v_axes.size();
  ge::TensorDesc tensorDesc;
  std::vector<int64_t> dims = {};
  if (num != 0) {
    dims.push_back(num);
  }
  ge::Shape shape(dims);
  tensorDesc.SetShape(shape);
  tensorDesc.SetDataType(DT_INT32);

  ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(v_axes.data()), v_axes.size() * sizeof(int));
  op_dest.SetAttr("axes", tensor);
  op_dest.SetAttr("keep_dims", keep_dims_attr);

  return SUCCESS;
}

static Status ParseOpToGraphReduceSum(const Operator& op, Graph& graph)
{
  auto data0 = op::Data("data0").set_attr_index(0);
  ge::Tensor value;
  if (op.GetAttr("axes", value) != SUCCESS) {
      ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get value from op failed");
      return FAILED;
  }

  auto data1 = op::Const("data1").set_attr_value(value);
  auto reducesum = op::ReduceSum().set_input_x(data0).set_input_axes(data1);

  bool flag = false;
  if (op.GetAttr("keep_dims", flag) != SUCCESS) {
      ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get keep_dims from op failed");
      return FAILED;
  }
  reducesum.set_attr_keep_dims(flag);

  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  output_indexs.emplace_back(reducesum, vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

Status ParseParamsReduceSum13(const Message* op_src, ge::Operator& op_dest)
{
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
      ONNX_PLUGIN_LOGE("ReduceSum13", "Dynamic cast op_src to NodeProto failed.");
      return FAILED;
  }
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::13::ReduceSum");

  int input_size = node->input_size();
  bool keep_dims = true;
  int noop_with_empty_axes = 0;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
      keep_dims = (attr.i() == 1);
    } else if (attr.name() == "noop_with_empty_axes" && attr.type() == ge::onnx::AttributeProto::INT) {
      noop_with_empty_axes = attr.i();
    }
  }

  op_dest.SetAttr("input_size", input_size);
  op_dest.SetAttr("keep_dims", keep_dims);
  op_dest.SetAttr("noop_with_empty_axes", noop_with_empty_axes);
  return SUCCESS;
}

static Status ParseOpToGraphReduceSum13(const Operator& op, Graph& graph)
{
  auto data0 = op::Data("data0").set_attr_index(0);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr){
    ONNX_PLUGIN_LOGE("Resize", "Get op desc failed.");
    return FAILED;
  }
  bool flag = false;
  if (op.GetAttr("keep_dims", flag) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get keep_dims from op failed");
    return FAILED;
  }
  int input_num = 1;
  if (op.GetAttr("input_size", input_num) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get input_num from op failed");
    return FAILED;
  }
  int empty_axes = 0;
  if (op.GetAttr("noop_with_empty_axes", empty_axes) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get attribute noop_with_empty_axes failed");
    return FAILED;
  }
  if (input_num == 1 && empty_axes == 0) {
    std::vector<int64_t> v_axes={};
    ge::TensorDesc tensorDesc;
    std::vector<int64_t> dims = {};
    ge::Shape shape(dims);
    tensorDesc.SetShape(shape);
    tensorDesc.SetDataType(DT_INT64);
    ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(v_axes.data()), v_axes.size() * sizeof(int));
    auto axes = op::Const("axes").set_attr_value(tensor);
    std::vector<Operator> inputs{data0};
    std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
    auto reducesum = op::ReduceSum()
                      .set_input_x(data0)
                      .set_input_axes(axes)
                      .set_attr_keep_dims(flag);
    output_indexs.emplace_back(reducesum, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
  } else if (input_num == 2 && empty_axes != 0) {
    auto data1 = op::Data("data1").set_attr_index(1);
    auto reducesum13 = op::ReduceSum()
                        .set_input_x(data0)
                        .set_input_axes(data1)
                        .set_attr_keep_dims(flag);
    std::vector<Operator> inputs{data0, data1};
    std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
    output_indexs.emplace_back(reducesum13, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
  } else {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "Input num or set attr is error");
    return FAILED;
  }
  return SUCCESS;
}

// register ReduceSum op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::ReduceSum",
                 "ai.onnx::9::ReduceSum",
                 "ai.onnx::10::ReduceSum",
                 "ai.onnx::11::ReduceSum",
                 "ai.onnx::12::ReduceSum"})
  .ParseParamsFn(parse_params_reduce_sum)
  .ParseOpToGraphFn(ParseOpToGraphReduceSum)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("ReduceSum")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::13::ReduceSum")
  .ParseParamsFn(ParseParamsReduceSum13)
  .ParseOpToGraphFn(ParseOpToGraphReduceSum13)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
