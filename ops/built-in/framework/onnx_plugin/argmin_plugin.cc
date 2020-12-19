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
Status ParseParamsArgMin(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("ArgMin", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  bool has_axis = false;

  // update attribute dimension from given attributes
  for (auto attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      op_dest.SetAttr("dimension", attr.i());
      has_axis = true;
      OP_LOGI("ArgMin", "set dimension = %d", attr.i());
    }
  }

  if (!has_axis) {
    op_dest.SetAttr("dimension", (int64_t)0);
    OP_LOGI("ArgMin", "use default dimension = 0.");
  }
  return SUCCESS;
}

// register ArgMin op info to GE
REGISTER_CUSTOM_OP("ArgMinD")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::ArgMin")
  .ParseParamsFn(ParseParamsArgMin)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
