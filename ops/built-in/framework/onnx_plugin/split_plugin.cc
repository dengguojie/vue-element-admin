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
#include "onnx_common.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
Status ParseParamsSplitNew(const Message *op_src, ge::Operator &op_dest) {
  const ge::onnx::NodeProto *node = dynamic_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(),  "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int axis = 0;
  std::vector<int64_t> split;
  for (auto attr : node->attribute()) {
    if (attr.name() == "axis") {
      axis = attr.i();
    } else if (attr.name() == "split") {
      for (auto dim : attr.ints()) {
        split.push_back(dim);
      } 
    }
  }

  int input_size = node->input_size();
  int output_size = node->output_size();
  int split_size = (int)split.size();
  if (split_size != 0 && split_size != output_size) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(),  "output size[%d] is not equal split size[%d].",
                     output_size, split_size);
    return FAILED;
  }
  op_dest.SetAttr("split_dim", axis);
  op_dest.SetAttr("size_splits", split);
  
  op_dest.SetAttr("input_size", input_size);
  op_dest.SetAttr("num_split", output_size);

  std::shared_ptr<ge::OpDesc> opDesc =
      ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (opDesc == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(),  "ParserParam get op desc failed.");
    return FAILED;
  }
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::13::Split");
  opDesc->AddDynamicInputDesc("x", input_size);
  opDesc->AddDynamicOutputDesc("y", output_size);
  return SUCCESS;
}

Status ParseOpToGraphSplitNew(const ge::Operator &op, Graph &graph) {
  auto input_x_0 = op::Data("x").set_attr_index(0);
  int32_t split_dim = 0;
  if (op.GetAttr("split_dim", split_dim) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(),  "get attr split_dim from op failed!.");
    return FAILED;
  }
  int num_split = 0;
  if (op.GetAttr("num_split", num_split) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(),  "get attr num_split from op failed!.");
    return FAILED;
  }
  std::vector<std::size_t> idx;
  for (std::size_t i = 0; i < (std::size_t)num_split; i++) {
    idx.push_back(i);
  }
  int input_size = 1;
  if (op.GetAttr("input_size", input_size) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(),  "get attr num_split from op failed!.");
    return FAILED;
  }

  std::vector<int64_t> split_dim_dims = {1};
  ge::Tensor split_dim_tensor = Scalar2Tensor(split_dim, split_dim_dims, ge::DT_INT32);
  auto const_split_dim_2 = op::Const("split_dim").set_attr_value(split_dim_tensor);

  if (input_size == 1) {
    std::vector<int64_t> size_splits;
    if (op.GetAttr("size_splits", size_splits) != SUCCESS) {
      ONNX_PLUGIN_LOGE(op.GetName().c_str(),  "get attr size_splits from op failed!.");
      return FAILED;
    }
    ge::Operator split_op;
    if (size_splits.empty()) {
      split_op = op::Split().create_dynamic_output_y(num_split)
                            .set_input_x(input_x_0)
                            .set_input_split_dim(const_split_dim_2)
                            .set_attr_num_split(num_split);
    } else {
      std::vector<int64_t> dims = {(int64_t)size_splits.size()};
      ge::Tensor size_splits_tensor = Vec2Tensor(size_splits, dims, ge::DT_INT64);
      auto const_size_splits = op::Const("size_splits").set_attr_value(size_splits_tensor);

      split_op = op::SplitV().create_dynamic_output_y(num_split)
                                  .set_input_x(input_x_0)
                                  .set_input_size_splits(const_size_splits)
                                  .set_input_split_dim(const_split_dim_2)
                                  .set_attr_num_split(num_split);
    }
   
    
    std::vector<ge::Operator> inputs = { input_x_0 };
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(split_op, idx);
    graph.SetInputs(inputs).SetOutputs(output_indexs);
  } else {
    auto input_size_splits_1 = op::Data("size_splits").set_attr_index(1);
    ge::TensorDesc splitDimTensorDesc;

    auto split_v = op::SplitV().create_dynamic_output_y(num_split)
                               .set_input_x(input_x_0)
                               .set_input_size_splits(input_size_splits_1)
                               .set_input_split_dim(const_split_dim_2)
                               .set_attr_num_split(num_split);
   
    std::vector<ge::Operator> inputs { input_x_0, input_size_splits_1 };
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(split_v, idx);
    graph.SetInputs(inputs).SetOutputs(output_indexs);
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::Split",
                 "ai.onnx::9::Split",
                 "ai.onnx::10::Split",
                 "ai.onnx::11::Split",
                 "ai.onnx::12::Split",
                 "ai.onnx::13::Split"})
  .ParseParamsFn(ParseParamsSplitNew)
  .ParseOpToGraphFn(ParseOpToGraphSplitNew)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
