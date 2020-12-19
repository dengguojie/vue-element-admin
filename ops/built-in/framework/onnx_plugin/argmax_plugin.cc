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

#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsArgMax(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("ArgMax", "Start into the ParseParamsArgMax!");
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("ArgMax", "Dynamic cast op_src to NodeProto failed");
    return FAILED;
  }

  int axis = 0;
  int keep_dims = 1;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis = attr.i();
    }
    if (attr.name() == "keepdims" &&
        attr.type() == ge::onnx::AttributeProto::INT) {
      keep_dims = attr.i();
    }
  }
  op_dest.SetAttr("dimension", axis);
  op_dest.SetAttr("keep_dims", keep_dims);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("ArgMaxD")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::ArgMax")
  .ParseParamsFn(ParseParamsArgMax)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
