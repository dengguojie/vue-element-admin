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

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {

Status ParseParamsClipV9(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE("Clip", "Dynamic cast op_src to NodeProto failed.");
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

  // 3.set attr if needed
  ge::TensorDesc tensorDesc;
  std::vector<int64_t> dims = {1};
  ge::Shape shape(dims);
  tensorDesc.SetShape(shape);
  tensorDesc.SetDataType(DT_FLOAT);

  ge::Tensor tensor1(tensorDesc, reinterpret_cast<uint8_t*>(&max), sizeof(float));
  op_dest.SetAttr("max", tensor1);
  ge::Tensor tensor2(tensorDesc, reinterpret_cast<uint8_t*>(&min), sizeof(float));
  op_dest.SetAttr("min", tensor2);

  return SUCCESS;
}

static Status ParseOpToGraphClipV9(const Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);
  ge::Tensor value1;
  if (op.GetAttr("max", value1) != SUCCESS) {
    ONNX_PLUGIN_LOGE("Clip", "get max from op failed");
    return FAILED;
  }
  ge::Tensor value2;
  if (op.GetAttr("min", value2) != SUCCESS) {
    ONNX_PLUGIN_LOGE("Clip", "get max from op failed");
    return FAILED;
  }

  auto data1 = op::Const("data1").set_attr_value(value1);
  auto data2 = op::Const("data2").set_attr_value(value2);
  auto cast_op = op::Cast("cast").set_input_x(data0).set_attr_dst_type(DT_FLOAT);
  auto clip_by_value =
      op::ClipByValue().set_input_x(cast_op).set_input_clip_value_min(data2).set_input_clip_value_max(data1);

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
    ONNX_PLUGIN_LOGE("Clip", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  bool no_max = false;
  bool no_min = false;
  int num = node->input_size();
  if (num < 2) {
    ONNX_PLUGIN_LOGE("Clip", "At least of 'min' or 'max' must not be None");
    return FAILED;
  } else if (num == 2) {
    no_min = false;
    no_max = true;
  } else {
    no_min = (node->input(1) == "");
    no_max = (node->input(2) == "");
  }

  op_dest.SetAttr("no_max", no_max);
  op_dest.SetAttr("no_min", no_min);

  return SUCCESS;
}

static Status ParseOpToGraphClipV11(const Operator& op, Graph& graph) {
  auto data0 = op::Data("x").set_attr_index(0);
  bool no_max = true;
  if (op.GetAttr("no_max", no_max) != SUCCESS) {
    ONNX_PLUGIN_LOGE("Clip", "get no_max from op failed");
    return FAILED;
  }
  bool no_min = true;
  if (op.GetAttr("no_min", no_min) != SUCCESS) {
    ONNX_PLUGIN_LOGE("Clip", "get no_min from op failed");
    return FAILED;
  }

  int index = 1;
  Operator min_op;
  if (no_min) {
    ONNX_PLUGIN_LOGI("Clip", "Min is ommitted, use default value=-3.402823e+38");
    float min = -3.402823e+38;
    ge::Shape shape({1});
    TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor min_tensor(tensorDesc, reinterpret_cast<uint8_t*>(&min), sizeof(ge::DT_FLOAT));
    min_op = op::Const("min").set_attr_value(min_tensor);
  } else {
    min_op = op::Data("min").set_attr_index(index);
  }
  index++;

  Operator max_op;
  if (no_max) {
    ONNX_PLUGIN_LOGI("Clip", "Max is ommitted, use default value=3.402823e+38");
    float max = 3.402823e+38;
    ge::Shape shape({1});
    TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor max_tensor(tensorDesc, reinterpret_cast<uint8_t*>(&max), sizeof(ge::DT_FLOAT));
    max_op = op::Const("max").set_attr_value(max_tensor);
  } else {
    max_op = op::Data("max").set_attr_index(index);
  }
  index++;

  Operator input_op1 = data0;
  Operator input_op2 = min_op;
  Operator input_op3 = max_op;
  auto clip_by_value = op::ClipByValue();
  if (no_max || no_min) {
    input_op1 = op::Cast("cast").set_input_x(data0).set_attr_dst_type(DT_FLOAT);

    if (!no_max) {
      input_op3 = op::Cast("cast1").set_input_x(max_op).set_attr_dst_type(DT_FLOAT);
    }

    if (!no_min) {
      input_op2 = op::Cast("cast2").set_input_x(min_op).set_attr_dst_type(DT_FLOAT);
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
    .OriginOpType({"ai.onnx::11::Clip", "ai.onnx::12::Clip", "ai.onnx::13::Clip"})
    .ParseParamsFn(ParseParamsClipV11)
    .ParseOpToGraphFn(ParseOpToGraphClipV11)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
