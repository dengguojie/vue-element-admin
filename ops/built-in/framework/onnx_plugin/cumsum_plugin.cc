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
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsCumSum(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("CumSum", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int exclusive = 0;
  int reverse = 0;
  bool exclusive_flag = false;
  bool reverse_flag = false;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "exclusive" &&
        attr.type() == ge::onnx::AttributeProto::INT) {
      exclusive = attr.i();
      if (exclusive == 1) {
        exclusive_flag = true;
      }
    } else if (attr.name() == "reverse" &&
               attr.type() == ge::onnx::AttributeProto::INT) {
      reverse = attr.i();
      if (reverse == 1) {
        reverse_flag = true;
      }
    }
  }
  op_dest.SetAttr("exclusive", exclusive_flag);
  op_dest.SetAttr("reverse", reverse_flag);

  return SUCCESS;
}

// register CumSum op info to GE
REGISTER_CUSTOM_OP("Cumsum")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::CumSum")
  .ParseParamsFn(ParseParamsCumSum)
  .ImplyType(ImplyType::TVM);
}  // namespace domi