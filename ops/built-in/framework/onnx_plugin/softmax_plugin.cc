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
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsSoftmax(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("Softmax", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  std::vector<int> v_axis;
  bool set_axes_flag = false;

  for (const auto &attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      v_axis.push_back(attr.i());
      set_axes_flag = true;
    }
  }
  if (!set_axes_flag) {
    // default value in version 9,11,12.
    v_axis.push_back(1);
  }
  op_dest.SetAttr("axes", v_axis);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("SoftmaxV2")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::9::Softmax",
                 "ai.onnx::10::Softmax",
                 "ai.onnx::11::Softmax",
                 "ai.onnx::12::Softmax"})
  .ParseParamsFn(ParseParamsSoftmax)
  .ImplyType(ImplyType::TVM);
}  // namespace domi