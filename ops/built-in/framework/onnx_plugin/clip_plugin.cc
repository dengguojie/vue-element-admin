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
 * \file clip_plugin.cpp
 * \brief
 */
#include "onnx_common.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
using namespace std;
using namespace ge;
using ge::Operator;

namespace {
  constexpr int ATTR_NUM = 2;
  constexpr int NO_MIN_IDX = 1;
  constexpr int NO_MAX_IDX = 2;
}

namespace domi {

Status ParseParamsClipV9(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);

  // 1.add dynamic input and out
  opDesc->AddDynamicInputDesc("x", 3);
  opDesc->AddDynamicOutputDesc("output", 1);

  // 2.set original_type
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::9::Clip");

  // attr max and min default value
  float max = 3.402823e+38;
  float min = -3.402823e+38;

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "max" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      max = attr.f();
    } else if (attr.name() == "min" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      min = attr.f();
    }
  }

  std::vector<int64_t> dims = {1};
  ge::Tensor tensor1 = Scalar2Tensor(max, dims, ge::DT_FLOAT);
  op_dest.SetAttr("max", tensor1);
  ge::Tensor tensor2 = Scalar2Tensor(min, dims, ge::DT_FLOAT);
  op_dest.SetAttr("min", tensor2);
  op_dest.SetAttr("name", node->name());

  return SUCCESS;
}

static Status ParseOpToGraphClipV9(const Operator& op, Graph& graph) {
  std::string ori_name;
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get name from op failed.");
    return FAILED;
  }
  auto data0 = op::Data(ori_name + "data0").set_attr_index(0);
  ge::Tensor value1;
  op.GetAttr("max", value1);
  ge::Tensor value2;
  op.GetAttr("min", value2);

  auto data1 = op::Const(ori_name + "data1").set_attr_value(value1);
  auto data2 = op::Const(ori_name + "data2").set_attr_value(value2);
  auto cast_op = op::Cast(ori_name + "cast").set_input_x(data0).set_attr_dst_type(DT_FLOAT);
  auto clip_by_value =
      op::ClipByValue(ori_name + "ClipByValue").set_input_x(cast_op)
                                               .set_input_clip_value_min(data2)
                                               .set_input_clip_value_max(data1);

  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  output_indexs.emplace_back(clip_by_value, vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

Status ParseParamsClipV11(const Message* op_src, ge::Operator& op_dest) {
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);

  // 1.add dynamic input and out
  opDesc->AddDynamicInputDesc("x", 3);
  opDesc->AddDynamicOutputDesc("output", 1);

  // 2.set original_type
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::11::Clip");

  // 3.set attr if needed
  const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  bool no_max = false;
  bool no_min = false;
  int num = node->input_size();
  if (num < ATTR_NUM) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "At least of 'min' or 'max' must not be None");
    return FAILED;
  } else if (num == ATTR_NUM) {
    no_min = false;
    no_max = true;
  } else {
    no_min = (node->input(NO_MIN_IDX) == "");
    no_max = (node->input(NO_MAX_IDX) == "");
  }

  op_dest.SetAttr("no_max", no_max);
  op_dest.SetAttr("no_min", no_min);
  op_dest.SetAttr("name", node->name());

  return SUCCESS;
}

static Status ParseOpToGraphClipV11(const Operator& op, Graph& graph) {
  std::string ori_name;
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get name from op failed.");
    return FAILED;
  }
  auto data0 = op::Data(ori_name + "x").set_attr_index(0);
  bool no_max = true;
  op.GetAttr("no_max", no_max);
  bool no_min = true;
  op.GetAttr("no_min", no_min);
  
  int index = 1;
  Operator min_op;
  if (no_min) {
    float min = -3.402823e+38;
    std::vector<int64_t> dims = {1};
    ge::Tensor min_tensor = Scalar2Tensor(min, dims, ge::DT_FLOAT);
    min_op = op::Const(ori_name + "min").set_attr_value(min_tensor);
  } else {
    min_op = op::Data(ori_name + "min").set_attr_index(index);
  }
  index++;

  Operator max_op;
  if (no_max) {
    float max = 3.402823e+38;
    std::vector<int64_t> dims = {1};
    ge::Tensor max_tensor = Scalar2Tensor(max, dims, ge::DT_FLOAT);
    max_op = op::Const(ori_name + "max").set_attr_value(max_tensor);
  } else {
    max_op = op::Data(ori_name + "max").set_attr_index(index);
  }
  index++;

  Operator input_op1 = data0;
  Operator input_op2 = min_op;
  Operator input_op3 = max_op;
  auto clip_by_value = op::ClipByValue(ori_name + "ClipByValue");
  if (no_max || no_min) {
    input_op1 = op::Cast(ori_name + "cast").set_input_x(data0).set_attr_dst_type(DT_FLOAT);

    if (!no_max) {
      input_op3 = op::Cast(ori_name + "cast1").set_input_x(max_op).set_attr_dst_type(DT_FLOAT);
    }

    if (!no_min) {
      input_op2 = op::Cast(ori_name + "cast2").set_input_x(min_op).set_attr_dst_type(DT_FLOAT);
    }
  }
  clip_by_value.set_input_x(input_op1).set_input_clip_value_min(input_op2).set_input_clip_value_max(input_op3);
  std::vector<Operator> inputs{data0, min_op, max_op};
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  output_indexs.emplace_back(clip_by_value, vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

// register Clip op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Clip", "ai.onnx::9::Clip", "ai.onnx::10::Clip"})
    .ParseParamsFn(ParseParamsClipV9)
    .ParseOpToGraphFn(ParseOpToGraphClipV9)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::11::Clip", "ai.onnx::12::Clip", "ai.onnx::13::Clip",
                   "ai.onnx::14::Clip", "ai.onnx::15::Clip"})
    .ParseParamsFn(ParseParamsClipV11)
    .ParseOpToGraphFn(ParseOpToGraphClipV11)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
