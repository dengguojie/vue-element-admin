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

#include <vector>

#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"

namespace domi {
Status ParseParamsSplit(const Message *op_src, ge::Operator &op_dst) {
  const ge::onnx::NodeProto *node =
      reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("Split", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
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
  int split_num = 0;
  op_dst.GetAttr("num_split", split_num);
  std::shared_ptr<ge::OpDesc> opDesc =
      ge::OpDescUtils::GetOpDescFromOperator(op_dst);
  opDesc->AddDynamicOutputDesc("y", split_num);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("SplitVD")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::Split")
  .ParseParamsFn(ParseParamsSplit)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
