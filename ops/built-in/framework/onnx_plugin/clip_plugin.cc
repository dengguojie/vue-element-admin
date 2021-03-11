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
#include <string>
#include <vector>
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "all_ops.h"
#include "graph.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {

Status ParseParamsClipV9(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("Clip", "Dynamic cast op_src to NodeProto failed.");
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

static Status ParseOpToGraphClip(const Operator& op, Graph& graph)
{
  auto data0 = op::Data("data0").set_attr_index(0);
  ge::Tensor value1;
  if (op.GetAttr("max", value1) != SUCCESS) {
    OP_LOGE("Clip", "get max from op failed");
    return FAILED;
  }
  ge::Tensor value2;
  if (op.GetAttr("min", value2) != SUCCESS) {
    OP_LOGE("Clip", "get max from op failed");
    return FAILED;
  }
  
  auto data1 = op::Const("data1").set_attr_value(value1);
  auto data2 = op::Const("data2").set_attr_value(value2);
  auto clip_by_value = op::ClipByValue().set_input_x(data0).set_input_clip_value_min(data2)
                                        .set_input_clip_value_max(data1);
  
  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  output_indexs.emplace_back(clip_by_value, vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

Status ParseParamsClipV11(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("Clip", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  return SUCCESS;
}


// register Clip op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::Clip",
                   "ai.onnx::10::Clip"})
    .ParseParamsFn(ParseParamsClipV9)
    .ParseOpToGraphFn(ParseOpToGraphClip)
    .ImplyType(ImplyType::TVM);
  
REGISTER_CUSTOM_OP("ClipByValue")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::11::Clip",
                   "ai.onnx::12::Clip",
                   "ai.onnx::13::Clip"})
    .ParseParamsFn(ParseParamsClipV11)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
