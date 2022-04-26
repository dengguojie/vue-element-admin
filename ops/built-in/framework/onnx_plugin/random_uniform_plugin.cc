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
 * \file randomuniform.cpp
 * \brief
 */
#include "onnx_common.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "random_ops.h"

using namespace ge;
namespace domi {
Status ParseParamsRandomuniform(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  opDesc->AddDynamicInputDesc("x", 3);
  opDesc->AddDynamicOutputDesc("y", 1);
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::11::RandomUniform");

  float high = 1.0;
  float low = 0.0;
  std::vector<int> op_shape;
  int seed = 0;
  int dtype = 1;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "dtype" && attr.type() == ge::onnx::AttributeProto::INT) {
      dtype = attr.i();
    } else if (attr.name() == "high" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      high = attr.f();
    } else if (attr.name() == "low" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      low = attr.f();
    } else if (attr.name() == "seed" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      seed = (int)attr.f();
    } else if (attr.name() == "shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        op_shape.push_back(attr.ints(i));
      }
    }
  }

  int num = op_shape.size();
  std::vector<int64_t> dims = {num};
  ge::Tensor tensor = Vec2Tensor(op_shape, dims, ge::DT_INT32);

  op_dest.SetAttr("shape", tensor);
  op_dest.SetAttr("max", high);
  op_dest.SetAttr("min", low);
  op_dest.SetAttr("seed", seed);
  op_dest.SetAttr("dtype", dtype);
  return SUCCESS;
}

Status ParseOpToGraphRandomuniform(const ge::Operator &op, ge::Graph &graph) {
  ge::Tensor shape;
  if (op.GetAttr("shape", shape) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get shape from op failed");
    return FAILED;
  }
  auto data0 = op::Const("data0").set_attr_value(shape);

  float max_f = 0.0;
  if (op.GetAttr("max", max_f) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get max from op failed");
    return FAILED;
  }

  float min_f = 0.0;
  if (op.GetAttr("min", min_f) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get min from op failed");
    return FAILED;
  }

  int dtype = 1;
  if (op.GetAttr("dtype", dtype) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get dtype from op failed");
    return FAILED;
  }
  // cast output to dst_dtype(onnx : Ascend)
  // float32, float16, int32, int64
  std::map<int, int> kvlist = {{1, 0}, {10, 1}, {6, 3}, {2, 9}};
  if (kvlist.find(dtype) == kvlist.end()){
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "only support float32/float16/int32/int64, but got %d", dtype);
    return FAILED;
  }
  ge::DataType temp_type = GetOmDtypeFromOnnxDtype(dtype);
  int seed = 0;
  if (op.GetAttr("seed", seed) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get seed from op failed");
    return FAILED;
  }

  std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
  if (dtype == 6 || dtype == 2) {
    int32_t max = max_f;
    std::vector<int64_t> dims = {};
    ge::Tensor max_tensor = Scalar2Tensor(max, dims, temp_type);
    auto data1 = op::Const("data1").set_attr_value(max_tensor);

    int32_t min = min_f;
    ge::Tensor min_tensor = Scalar2Tensor(min, dims, temp_type);
    auto data2 = op::Const("data2").set_attr_value(min_tensor);

    auto random_int = op::RandomUniformInt()
                        .set_input_shape(data0)
                        .set_input_min(data2)
                        .set_input_max(data1)
                        .set_attr_seed(seed)
                        .set_attr_seed2(seed);
    std::vector<ge::Operator> inputs{data0, data1, data2};
    outputs.emplace_back(random_int, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
  } else {
    auto random = op::RandomUniform()
                    .set_input_shape(data0)
                    .set_attr_dtype(temp_type)
                    .set_attr_seed(seed)
                    .set_attr_seed2(seed);
    float mul_num = max_f - min_f;
    auto random_mul = op::Muls().set_input_x(random).set_attr_value(mul_num);
    auto random_float = op::Adds().set_input_x(random_mul).set_attr_value(min_f);
    std::vector<ge::Operator> inputs{data0};
    outputs.emplace_back(random_float, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
  }

  return SUCCESS;
}

// register Addcmul op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::RandomUniform",
                   "ai.onnx::9::RandomUniform",
                   "ai.onnx::10::RandomUniform",
                   "ai.onnx::11::RandomUniform",
                   "ai.onnx::12::RandomUniform",
                   "ai.onnx::13::RandomUniform"})
    .ParseParamsFn(ParseParamsRandomuniform)
    .ParseOpToGraphFn(ParseOpToGraphRandomuniform)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
