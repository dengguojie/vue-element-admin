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
#include "graph.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "all_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
Status ParseSplitAttr(const ge::onnx::NodeProto *node, ge::Operator &op_dst) {
  bool b_find_axis = false;
  bool b_find_split = false;
  int axis_val = 0;
  for (auto attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis_val = attr.i();
      op_dst.SetAttr("split_dim", axis_val);
      b_find_axis = true;
    } else if (attr.name() == "split" &&
               attr.type() == ge::onnx::AttributeProto::INTS) {
      std::vector<int64_t> split;
      for (auto axis : attr.ints()) {
        split.push_back(axis);
      }
      int split_num = split.size();
      op_dst.SetAttr("size_splits", split);
      op_dst.SetAttr("num_split", split_num);
      b_find_split = true;
    }
    if (b_find_axis && b_find_split) {
      break;
    }
  }
  if (!b_find_axis) {
    op_dst.SetAttr("split_dim", axis_val);
  }
  if (!b_find_split) {
    // acquire the Output number of operator
    int op_output_size = node->output_size();
    std::vector<int64_t> split_size;

    op_dst.SetAttr("size_splits", split_size);
    op_dst.SetAttr("num_split", op_output_size);
  }

  return SUCCESS;
}

Status ParseParamsSplit(const Message *op_src, ge::Operator &op_dst) {
  const ge::onnx::NodeProto *node = dynamic_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("Split", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  ParseSplitAttr(node, op_dst);

  int split_num = 0;
  op_dst.GetAttr("num_split", split_num);
  std::shared_ptr<ge::OpDesc> opDesc =
      ge::OpDescUtils::GetOpDescFromOperator(op_dst);
  if (opDesc == nullptr) {
    OP_LOGE("Split", "ParserParam get op desc failed.");
    return FAILED;
  }
  opDesc->AddDynamicOutputDesc("y", split_num);
  return SUCCESS;
}

Status ParseParamsSplitNew(const Message *op_src, ge::Operator &op_dst) {
  const ge::onnx::NodeProto *node = dynamic_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("Split", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  ParseSplitAttr(node, op_dst);

  int input_size = node->input_size();
  op_dst.SetAttr("input_size", input_size);
  input_size = (input_size == 1) ? 1 : 3;
  int output_size = 0;
  op_dst.GetAttr("num_split", output_size);

  std::shared_ptr<ge::OpDesc> opDesc =
      ge::OpDescUtils::GetOpDescFromOperator(op_dst);
  if (opDesc == nullptr) {
    OP_LOGE("Split", "ParserParam get op desc failed.");
    return FAILED;
  }
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::13::Split");
  opDesc->AddDynamicInputDesc("args", input_size);
  opDesc->AddDynamicOutputDesc("output", output_size);
  return SUCCESS;
}

Status ParseOpToGraphSplitNew(const ge::Operator &op, Graph &graph) {
  auto input_x_0 = op::Data("x").set_attr_index(0);
  int split_dim = 0;
  if (op.GetAttr("split_dim", split_dim) != SUCCESS) {
    OP_LOGE("PartitionedCall(Split)", "get attr split_dim from op failed!");
    return FAILED;
  }
  int num_split = 0;
  if (op.GetAttr("num_split", num_split) != SUCCESS) {
    OP_LOGE("PartitionedCall(Split)", "get attr num_split from op failed!");
    return FAILED;
  }
  std::vector<std::size_t> idx;
  for (std::size_t i = 0; i < (std::size_t)num_split; i++) {
    idx.push_back(i);
  }
  int input_size = 1;
  if (op.GetAttr("input_size", input_size) != SUCCESS) {
    OP_LOGE("PartitionedCall(Split)", "get attr num_split from op failed!");
    return FAILED;
  }

  if (input_size == 1) {
    std::vector<int64_t> attr_size_splits;
    if (op.GetAttr("size_splits", attr_size_splits) != SUCCESS) {
      OP_LOGE("PartitionedCall(SplitVD)", "get attr size_splits from op failed!");
      return FAILED;
    }
    auto split_v_d = op::SplitVD().set_input_x(input_x_0)
                                  .set_attr_size_splits(attr_size_splits)
                                  .set_attr_split_dim(split_dim)
                                  .set_attr_num_split(num_split);
    std::shared_ptr<ge::OpDesc> opDesc =
      ge::OpDescUtils::GetOpDescFromOperator(split_v_d);
    if (opDesc == nullptr) {
      OP_LOGE("SplitVD", "ParserOpToGraph get op desc failed.");
      return FAILED;
    }
    opDesc->AddDynamicOutputDesc("y", num_split);
    std::vector<ge::Operator> inputs = { input_x_0 };
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(split_v_d, idx);
    graph.SetInputs(inputs).SetOutputs(output_indexs);
  } else {
    auto input_size_splits_1 = op::Data("size_splits").set_attr_index(1);
    ge::TensorDesc splitDimTensorDesc;
    int split_dim_size = 1;
    std::vector<int64_t> split_dim_dims = {split_dim_size};
    ge::Shape splitDimShape(split_dim_dims);
    splitDimTensorDesc.SetShape(splitDimShape);
    splitDimTensorDesc.SetDataType(DT_INT32);
    ge::Tensor splitDimTensor(
      splitDimTensorDesc, reinterpret_cast<uint8_t *>(&split_dim), split_dim_size * sizeof(int32_t));
    auto const_split_dim_2 = op::Const("split_dim").set_attr_value(splitDimTensor);

    auto split_v = op::SplitV().set_input_x(input_x_0)
                               .set_input_size_splits(input_size_splits_1)
                               .set_input_split_dim(const_split_dim_2)
                               .set_attr_num_split(num_split);
    std::shared_ptr<ge::OpDesc> opDesc =
      ge::OpDescUtils::GetOpDescFromOperator(split_v);
    if (opDesc == nullptr) {
      OP_LOGE("SplitV", "ParserOpToGraph get op desc failed.");
      return FAILED;
    }
    opDesc->AddDynamicOutputDesc("y", num_split);
    std::vector<ge::Operator> inputs { input_x_0, input_size_splits_1 };
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(split_v, idx);
    graph.SetInputs(inputs).SetOutputs(output_indexs);
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("SplitVD")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::Split",
                 "ai.onnx::9::Split",
                 "ai.onnx::10::Split",
                 "ai.onnx::11::Split",
                 "ai.onnx::12::Split"})
  .ParseParamsFn(ParseParamsSplit)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::13::Split")
  .ParseParamsFn(ParseParamsSplitNew)
  .ParseOpToGraphFn(ParseOpToGraphSplitNew)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
