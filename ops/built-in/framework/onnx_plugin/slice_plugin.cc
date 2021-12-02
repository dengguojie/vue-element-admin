/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this
 * file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http:// www.apache.org/licenses/LICENSE-2.0
 */

#include "onnx_common.h"
#include "array_ops.h"
#include "selection_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;
Status ParseParamSlice(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status ParseParamSliceCall(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  std::vector<int64_t> ends = {};
  std::vector<int64_t> starts = {};
  std::vector<int64_t> axes = {};

  bool is_have_axes = false;
  for (auto attr : node->attribute()) {
    if (attr.name() == "ends") {
      int num = attr.ints_size();
      for (int i = 0; i < num; ++i) {
        ends.push_back(attr.ints(i));
      }
    } else if (attr.name() == "starts") {
      int num = attr.ints_size();
      for (int i = 0; i < num; ++i) {
        starts.push_back(attr.ints(i));
      }
    } else if (attr.name() == "axes") {
      is_have_axes = true;
      int num = attr.ints_size();
      for (int i = 0; i < num; ++i) {
        axes.push_back(attr.ints(i));
      }
      op_dest.SetAttr("axes", axes);
    }
  }
  if (ends.empty() || starts.empty()) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "attr must have ends and starts");
    return FAILED;
  }

  op_dest.SetAttr("end", ends);
  op_dest.SetAttr("begin", starts);
  op_dest.SetAttr("is_have_axes", is_have_axes);
  op_dest.SetAttr("original_type", "ai.onnx::9::Slice");
  OpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", 1);
  op_desc->AddDynamicOutputDesc("y", 1);
  return SUCCESS;
}

ge::Operator MakeConstOp(const ge::Operator& op, const std::string& attr_name) {
  std::vector<int64_t> val = {};
  op.GetAttr(attr_name, val);
  std::vector<int64_t> dims(1, val.size());
  ge::Tensor tensor = Vec2Tensor(val, dims, ge::DT_INT64);
  ge::Operator const_op = op::Const(attr_name).set_attr_value(tensor);
  return const_op;
}

Status ParseOpToGraphSlice(const ge::Operator& op, Graph& graph) {
  auto data_op = op::Data("input1").set_attr_index(0);
  auto const_op = MakeConstOp(op, "begin");
  auto const_op1 = MakeConstOp(op, "end");
  auto slice_op =
      op::StridedSliceV2("StridedSliceV2").set_input_x(data_op).set_input_begin(const_op).set_input_end(const_op1);

  bool is_have_axes = false;
  op.GetAttr("is_have_axes", is_have_axes);
  if (is_have_axes) {
    auto const_op2 = MakeConstOp(op, "axes");
    slice_op.set_input_axes(const_op2);
  }

  std::vector<ge::Operator> inputs{data_op};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(slice_op, std::vector<size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Slice","ai.onnx::9::Slice"})
    .ParseParamsFn(ParseParamSliceCall)
    .ParseOpToGraphFn(ParseOpToGraphSlice)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("StridedSliceV2")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::11::Slice", "ai.onnx::10::Slice", "ai.onnx::12::Slice", "ai.onnx::13::Slice"})
    .ParseParamsFn(ParseParamSlice)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
