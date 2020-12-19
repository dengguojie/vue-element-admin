/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Create: 2020-08-27
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
Status ParseParamsFlatten(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("Flatten", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  for (auto attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      if (attr.i() != 1) {
        OP_LOGE("Flatten", "axis value [%d] is not support.", attr.i());
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

// register Abs op info to GE
REGISTER_CUSTOM_OP("Flatten")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::Flatten")
  .ParseParamsFn(ParseParamsFlatten)
  .ImplyType(ImplyType::TVM);
}  // namespace domi