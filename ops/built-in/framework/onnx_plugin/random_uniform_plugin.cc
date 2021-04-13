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
#include <string>
#include <vector>

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "../../op_proto/inc/all_ops.h"
#include "op_log.h"
#include "operator.h"

using namespace ge;
namespace domi {
Status ParseParamsRandomuniform(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("Randomuniform", "Dynamic cast op_src to NodeProto failed.");
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
  ge::Shape shape({num});
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, ge::DT_INT32);
  ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(op_shape.data()), op_shape.size() * sizeof(ge::DT_INT32));

  op_dest.SetAttr("shape", tensor);
  op_dest.SetAttr("max", high);
  op_dest.SetAttr("min", low);
  op_dest.SetAttr("seed", seed);
  op_dest.SetAttr("dtype", dtype);
  return SUCCESS;
}

Status ParseOpToGraphRandomuniform(const ge::Operator &op, ge::Graph &graph) {
  float temp_num = 1000.;
  ge::Tensor shape;
  if (op.GetAttr("shape", shape) != SUCCESS) {
    OP_LOGE("Randomuniform", "get shape from op failed");
    return FAILED;
  }
  auto data0 = op::Const("data0").set_attr_value(shape);

  float max_f = 0.0;
  if (op.GetAttr("max", max_f) != SUCCESS) {
    OP_LOGE("Randomuniform", "get max from op failed");
    return FAILED;
  }
  int32_t max = max_f * temp_num;
  ge::TensorDesc tensorDesc1(ge::Shape(), FORMAT_ND, ge::DT_INT32);
  ge::Tensor max_tensor(tensorDesc1, reinterpret_cast<uint8_t*>(&max), sizeof(ge::DT_INT32));
  auto data1 = op::Const("data1").set_attr_value(max_tensor);

  float min_f = 0.0;
  if (op.GetAttr("min", min_f) != SUCCESS) {
    OP_LOGE("Randomuniform", "get min from op failed");
    return FAILED;
  }
  int32_t min = min_f * temp_num;
  ge::TensorDesc tensorDesc2(ge::Shape(), FORMAT_ND, ge::DT_INT32);
  ge::Tensor min_tensor(tensorDesc2, reinterpret_cast<uint8_t*>(&min), sizeof(ge::DT_INT32));
  auto data2 = op::Const("data2").set_attr_value(min_tensor);

  int seed = 0;
  if (op.GetAttr("seed", seed) != SUCCESS) {
    OP_LOGE("Randomuniform", "get seed from op failed");
    return FAILED;
  }

  auto random = op::RandomUniformInt()
                    .set_input_shape(data0)
                    .set_input_max(data1)
                    .set_input_min(data2)
                    .set_attr_seed(seed)
                    .set_attr_seed2(seed);

  auto random_fp32 = op::Cast().set_input_x(random).set_attr_dst_type(0);

  ge::TensorDesc tensorDesc3(ge::Shape({1}), FORMAT_ND, ge::DT_FLOAT);
  ge::Tensor temp_tensor(tensorDesc3, reinterpret_cast<uint8_t*>(&temp_num), sizeof(ge::DT_FLOAT));
  auto data3 = op::Const("data3").set_attr_value(temp_tensor);
  auto div = op::Div().set_input_x1(random_fp32).set_input_x2(data3);

  int dtype = 1;
  if (op.GetAttr("dtype", dtype) != SUCCESS) {
    OP_LOGE("Randomuniform", "get dtype from op failed");
    return FAILED;
  }
  // cast output to dst_dtype(onnx : Ascend)
  // float, float16, int8, int32, uint8, uint64, bool
  std::map<int, int> kvlist = {
      {1, 0}, {10, 1}, {3, 2}, {6, 3}, {2, 4}, {13, 10}, {9, 12}};
  if (kvlist.find(dtype) == kvlist.end()){
    OP_LOGE("Randomuniform", "only support float/float16/int8/int32/uint8/uint64/bool, but got %d", dtype);
    return FAILED;
  }
  auto ret = op::Cast().set_input_x(div).set_attr_dst_type(kvlist[dtype]);

  std::vector<ge::Operator> inputs{data0, data1, data2};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
  outputs.emplace_back(ret, std::vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

// register Addcmul op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::RandomUniform",
                   "ai.onnx::10::RandomUniform",
                   "ai.onnx::11::RandomUniform",
                   "ai.onnx::12::RandomUniform",
                   "ai.onnx::13::RandomUniform"})
    .ParseParamsFn(ParseParamsRandomuniform)
    .ParseOpToGraphFn(ParseOpToGraphRandomuniform)
    .ImplyType(ImplyType::TVM);
}  // namespace domi