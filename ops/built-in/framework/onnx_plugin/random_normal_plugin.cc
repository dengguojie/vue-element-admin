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
 * \file random_normal_plugin.cc
 * \brief
 */
#include "onnx_common.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "random_ops.h"

using namespace ge;
namespace domi {
Status ParseParamsRandomNormal(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  opDesc->AddDynamicInputDesc("x", 1);
  opDesc->AddDynamicOutputDesc("y", 1);
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::11::RandomNormal");

  int dtype = 1;
  float mean = 0.0f;
  float scale = 1.0f;
  int seed = 0;
  std::vector<int> shape_list;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "dtype" && attr.type() == ge::onnx::AttributeProto::INT) {
      dtype = attr.i();
    } else if (attr.name() == "mean" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      mean = attr.f();
    } else if (attr.name() == "scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      scale = attr.f();
    } else if (attr.name() == "seed" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      seed = (int)attr.f();
    } else if (attr.name() == "shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int dim : attr.ints()) {
        shape_list.push_back(dim);
      }
    };
  }
  if (shape_list.empty()) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Attr of shape must be not null.");
    return FAILED;
  }

  op_dest.SetAttr("mean", mean);
  op_dest.SetAttr("scale", scale);
  op_dest.SetAttr("seed", seed);
  op_dest.SetAttr("dtype", dtype);
  op_dest.SetAttr("shape", shape_list);
  return SUCCESS;
}

Status ParseOpToGraphRandomNormal(const ge::Operator& op, ge::Graph& graph) {
  float mean = 0.0f;
  op.GetAttr("mean", mean);

  float scale = 1.0f;
  op.GetAttr("scale", scale);

  int dtype = 1;
  op.GetAttr("dtype", dtype);

  std::vector<int> shape;
  op.GetAttr("shape", shape);

  if (shape.empty()) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "Attr of shape must be not null.");
    return FAILED;
  }

  // cast from onnx dtype to tbe dtype
  std::map<int, ge::DataType> kvlist = {{1, ge::DT_FLOAT}, {10, ge::DT_FLOAT16}, {11, ge::DT_DOUBLE}};
  if (kvlist.find(dtype) == kvlist.end()) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "only support float32/half/double, but got %d", dtype);
    return FAILED;
  }

  int seed = 0;
  op.GetAttr("seed", seed);

  ge::TensorDesc tensorDescInput;
  int64_t shapeLen = shape.size();
  std::vector<int64_t> dimsInput = {shapeLen};
  ge::Shape shapeInput(dimsInput);
  tensorDescInput.SetShape(shapeInput);
  tensorDescInput.SetDataType(ge::DT_INT32);

  ge::Tensor tensor_shape(tensorDescInput, reinterpret_cast<uint8_t*>(shape.data()), shapeLen * sizeof(int));

  auto shape_op = op::Const("input_shape").set_attr_value(tensor_shape);
  auto random_op = op::RandomStandardNormal("randomNormal")
                       .set_input_shape(shape_op)
                       .set_attr_dtype(kvlist[dtype])
                       .set_attr_seed(seed)
                       .set_attr_seed2(seed);

  auto mul_op = op::Muls("mul").set_input_x(random_op).set_attr_value(scale);
  auto add_op = op::Adds("add").set_input_x(mul_op).set_attr_value(mean);
  std::vector<ge::Operator> inputs{shape_op};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
  outputs.emplace_back(add_op, std::vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

// register Addcmul op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::RandomNormal", "ai.onnx::9::RandomNormal", "ai.onnx::10::RandomNormal",
                   "ai.onnx::11::RandomNormal", "ai.onnx::12::RandomNormal", "ai.onnx::13::RandomNormal"})
    .ParseParamsFn(ParseParamsRandomNormal)
    .ParseOpToGraphFn(ParseOpToGraphRandomNormal)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
