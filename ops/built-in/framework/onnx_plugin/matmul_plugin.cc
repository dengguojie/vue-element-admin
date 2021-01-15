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
Status ParseParamsMatMul(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("MatMul", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  bool trans_a = false;
  bool trans_b = false;

  op_dest.SetAttr("adj_x1", trans_a);
  op_dest.SetAttr("adj_x2", trans_b);

  return SUCCESS;
}

// register MatMul op info to GE
REGISTER_CUSTOM_OP("BatchMatMul")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::MatMul")
    .ParseParamsFn(ParseParamsMatMul)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
