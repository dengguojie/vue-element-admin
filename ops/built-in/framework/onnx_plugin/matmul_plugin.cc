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
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("MatMul", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  bool trans_a = false;
  bool trans_b = false;

  op_dest.SetAttr("adj_x1", trans_a);
  op_dest.SetAttr("adj_x2", trans_b);

  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (op_dsc == nullptr) {
    OP_LOGE("MatMul", "Get op desc failed. " );
    return FAILED;
  }

  ge::GeTensorDesc org_tensor_y = op_dsc->GetOutputDesc(0);
  org_tensor_y.SetOriginFormat(ge::FORMAT_NCHW);
  org_tensor_y.SetFormat(ge::FORMAT_NCHW);
  auto ret_y = op_dsc->UpdateOutputDesc(0, org_tensor_y);
  if (ret_y != ge::GRAPH_SUCCESS) {
    OP_LOGE("MatMul", "Update output format failed. " );
    return FAILED;
  }
  return SUCCESS;
}

// register MatMul op info to GE
REGISTER_CUSTOM_OP("BatchMatMulV2")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::MatMul",
                   "ai.onnx::9::MatMul",
                   "ai.onnx::10::MatMul",
                   "ai.onnx::11::MatMul",
                   "ai.onnx::12::MatMul",
                   "ai.onnx::13::MatMul"})
    .ParseParamsFn(ParseParamsMatMul)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
