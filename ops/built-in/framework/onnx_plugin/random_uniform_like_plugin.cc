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
 * \file random_normal_like.cc
 * \brief
 */
#include "onnx_common.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "random_ops.h"

using namespace ge;
namespace domi {
Status ParseParamsRandomUniformLike(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  opDesc->AddDynamicInputDesc("x", 1);
  opDesc->AddDynamicOutputDesc("y", 1);
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::11::RandomUniformLike");

  int dtype = 1;
  float low = 0.0f;
  float high = 1.0f;
  int seed = 0;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "dtype") {
      dtype = attr.i();
    } else if (attr.name() == "high") {
      high = attr.f();
    } else if (attr.name() == "low") {
      low = attr.f();
    } else if (attr.name() == "seed") {
      seed = (int)attr.f();
    }
  }

  op_dest.SetAttr("high", high);
  op_dest.SetAttr("low", low);
  op_dest.SetAttr("seed", seed);
  op_dest.SetAttr("dtype", dtype);
  return SUCCESS;
}

Status ParseOpToGraphRandomUniformLike(const ge::Operator &op, ge::Graph &graph) {
  float low = 0.0f;
  op.GetAttr("low", low);

  float high = 1.0f;
  op.GetAttr("high", high);

  int dtype = 1;
  op.GetAttr("dtype", dtype);

  // cast from onnx dtype to tbe dtype
  std::map<int, ge::DataType> kvlist = {{1, ge::DT_FLOAT}, {10, ge::DT_FLOAT16}, {11, ge::DT_DOUBLE}};
  if (kvlist.find(dtype) == kvlist.end()){
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "only support float32/float16/double, but got %d", dtype);
    return FAILED;
  }
  
  int seed = 0;
  op.GetAttr("seed", seed);
  
  std::vector<int64_t> dims = {1};
  float delta = high - low;
  ge::Tensor tensor_mean = Scalar2Tensor(low, dims, ge::DT_FLOAT);
  ge::Tensor tensor_scale = Scalar2Tensor(delta, dims, ge::DT_FLOAT);
  
  auto data0 = op::Data("data0").set_attr_index(0);
  auto shape_op = op::Shape("shape").set_input_x(data0).set_attr_dtype(ge::DT_INT32);
  auto random_op = op::RandomUniform("random_uniform").set_input_shape(shape_op)
                                                                     .set_attr_dtype(kvlist[dtype])
                                                                     .set_attr_seed(seed)
                                                                     .set_attr_seed2(seed);
  auto const_scale = op::Const("const_scale").set_attr_value(tensor_scale);
  auto const_mean = op::Const("const_mean").set_attr_value(tensor_mean);
  auto cast_mean = op::Cast("cast_mean").set_input_x(const_mean).set_attr_dst_type(kvlist[dtype]);
  auto cast_scale = op::Cast("cast_scale").set_input_x(const_scale).set_attr_dst_type(kvlist[dtype]);

  auto mul_op = op::Mul("mul").set_input_x1(random_op).set_input_x2(cast_scale);
  auto add_op = op::Add("add").set_input_x1(mul_op).set_input_x2(cast_mean);
  std::vector<ge::Operator> inputs{data0};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
  outputs.emplace_back(add_op, std::vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

// register Addcmul op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::RandomUniformLike",
                   "ai.onnx::9::RandomUniformLike",
                   "ai.onnx::10::RandomUniformLike",
                   "ai.onnx::11::RandomUniformLike",
                   "ai.onnx::12::RandomUniformLike",
                   "ai.onnx::13::RandomUniformLike"})
    .ParseParamsFn(ParseParamsRandomUniformLike)
    .ParseOpToGraphFn(ParseOpToGraphRandomUniformLike)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
