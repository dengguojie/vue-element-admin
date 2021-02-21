/* Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <string>
#include <vector>

#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph.h"
#include "all_ops.h"

using namespace ge;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsReduceMean(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("ReduceMean", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::vector<int> axes = {};
  bool keep_dims = true;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        axes.push_back(attr.ints(i));
      }
    } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
      keep_dims = (attr.i() == 1);
    }
  }
  int num = axes.size();
  std::vector<int64_t> dims = {num};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_NCHW, DT_INT32);
  ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(axes.data()), num * sizeof(DT_INT32));

  op_dest.SetAttr("axes", tensor);
  op_dest.SetAttr("keep_dims", keep_dims);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (op_desc == nullptr) {
    OP_LOGE("ReduceMean", "Get OpDesc from operator failed.");
    return FAILED;
  }
  op_desc->AddDynamicInputDesc("x", 2);
  op_desc->AddDynamicOutputDesc("y", 1);
  op_dest.SetAttr("original_type", "ai.onnx::11::ReduceMean");

  return SUCCESS;
}

static Status ParseOpToGraphReduceMean(const ge::Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);

  ge::Tensor axes;
  if (op.GetAttr("axes", axes) != SUCCESS) {
    OP_LOGE("ReduceMean", "get axes from op failed");
    return FAILED;
  }
  auto data1 = op::Const("data1").set_attr_value(axes);
  auto Reducemean = op::ReduceMean().set_input_x(data0).set_input_axes(data1);

  bool keep_dims = false;
  if (op.GetAttr("keep_dims", keep_dims) != SUCCESS) {
    OP_LOGE("ReduceMean", "get keep_dims from op failed");
    return FAILED;
  }
  Reducemean.set_attr_keep_dims(keep_dims);

  std::vector<ge::Operator> inputs{data0};
  std::vector<std::pair<ge::Operator, std::vector<size_t> > > outputs;
  outputs.emplace_back(Reducemean, std::vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(outputs);

  return SUCCESS;
}

// register ReduceMean op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::ReduceMean")
    .ParseParamsFn(ParseParamsReduceMean)
    .ParseOpToGraphFn(ParseOpToGraphReduceMean)
    .ImplyType(ImplyType::TVM);
}  // namespace domi