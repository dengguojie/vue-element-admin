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

#include <vector>
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamSlice(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("Slice", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status ParseParamSliceV9(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("SliceV9", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  std::vector<int> ends;
  std::vector<int> starts;
  bool is_have_ends = false;
  bool is_have_starts = false;
  for (auto attr : node->attribute()) {
    if (attr.name() == "ends") {
      is_have_ends = true;
      int num = attr.ints_size();
      for (int i = 0; i < num; ++i) {
        ends.push_back(attr.ints(i));
      }
    } else if (attr.name() == "starts") {
      is_have_starts = true;
      int num = attr.ints_size();
      for (int i = 0; i < num; ++i) {
        starts.push_back(attr.ints(i));
      }
    }
  }
  if (!is_have_starts || !is_have_ends) {
    OP_LOGE("SliceV9", "attr must have ends and starts");
    return FAILED;
  }
  int size = starts.size();
  std::vector<int> strides(size, 1);
  op_dest.SetAttr("end", ends);
  op_dest.SetAttr("begin", starts);
  op_dest.SetAttr("strides", strides);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("StridedSliceD")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::9::Slice")
  .ParseParamsFn(ParseParamSliceV9)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("StridedSliceV2")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::Slice")
  .ParseParamsFn(ParseParamSlice)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("StridedSliceV2")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::12::Slice")
  .ParseParamsFn(ParseParamSlice)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("StridedSliceV2")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::13::Slice")
  .ParseParamsFn(ParseParamSlice)
  .ImplyType(ImplyType::TVM);
}  // namespace domi