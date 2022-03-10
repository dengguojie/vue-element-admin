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
#include "array_ops.h"
#include "../../op_proto/util/axis_util.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsMatMul(const Message* op_src, ge::Operator& op_dest)
{
  ge::AscendString op_name;
  CHECK(op_dest.GetName(op_name) != ge::GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return FAILED);
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  CHECK(node == nullptr, ONNX_PLUGIN_LOGE(op_name.GetString(), "Dynamic cast op_src to NodeProto failed."),
        return FAILED);
// add the attr to support the custom matmul transpose fusion
  bool trans_a = false;
  bool trans_b = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "transA" && attr.i() != 0) {
      trans_a = true;
    }
    if (attr.name() == "transB" && attr.i() != 0) {
      trans_b = true;
    }
  }
  
  op_dest.SetAttr("adj_x1", trans_a);
  op_dest.SetAttr("adj_x2", trans_b);

  if (ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, false) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

// register MatMul op info to GE
REGISTER_CUSTOM_OP("BatchMatMulV2")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::MatMul"),
                   ge::AscendString("ai.onnx::9::MatMul"),
                   ge::AscendString("ai.onnx::10::MatMul"),
                   ge::AscendString("ai.onnx::11::MatMul"),
                   ge::AscendString("ai.onnx::12::MatMul"),
                   ge::AscendString("ai.onnx::13::MatMul")})
    .ParseParamsFn(ParseParamsMatMul)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
