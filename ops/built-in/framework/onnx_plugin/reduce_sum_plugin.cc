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

Status parse_params_reduce_sum(const Message* op_src, ge::Operator& op_dest)
{
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
      OP_LOGE("ReduceSum", "Dynamic cast op_src to NodeProto failed.");
      return FAILED;
  }

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);

  // 1.add dynamic input and out
  opDesc->AddDynamicInputDesc("x", 2);
  opDesc->AddDynamicOutputDesc("output", 1);

  // 2.set original_type
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::11::ReduceSum");

  std::vector<int> v_axes;
  bool keep_dims = true;

  for (const auto& attr : node->attribute()) {
      if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
      // std::copy(attr.ints().begin(), attr.ints().end(), v_axes.begin());
          for (int i = 0; i<attr.ints_size(); i++){
              v_axes.push_back(attr.ints(i));
          }
      }
      else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
          if (attr.i() != 1) {
              keep_dims = true;
          }
      }
  }

  // 3.set attr if needed
  int num = v_axes.size();
  ge::TensorDesc tensorDesc;
  std::vector<int64_t> dims = {num};
  ge::Shape shape(dims);
  tensorDesc.SetShape(shape);
  tensorDesc.SetDataType(DT_INT32);

  ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(v_axes.data()), v_axes.size() * sizeof(int));
  op_dest.SetAttr("axes", tensor);
  op_dest.SetAttr("keep_dims", keep_dims);

  return SUCCESS;
}

static Status ParseOpToGraphReduceSum(const Operator& op, Graph& graph)
{
  auto data0 = op::Data("data0").set_attr_index(0);
  ge::Tensor value;
  if (op.GetAttr("axes", value) != SUCCESS) {
      OP_LOGE("Reducesum", "get value from op failed");
      return FAILED;
  }

  auto data1 = op::Const("data1").set_attr_value(value);
  auto reducesum = op::ReduceSum().set_input_x(data0).set_input_axes(data1);

  bool flag = false;
  if (op.GetAttr("keep_dims", flag) != SUCCESS) {
      OP_LOGE("Reducesum", "get keep_dims from op failed");
      return FAILED;
  }
  reducesum.set_attr_keep_dims(flag);

  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  output_indexs.emplace_back(reducesum, vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

// register ReduceSum op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::ReduceSum")
  .ParseParamsFn(parse_params_reduce_sum)
  .ParseOpToGraphFn(ParseOpToGraphReduceSum)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
