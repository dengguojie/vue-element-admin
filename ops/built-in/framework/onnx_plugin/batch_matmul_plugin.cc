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
#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsBatchMatMul(const Message *opSrc, ge::Operator &opDst) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(opSrc);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(opDst.GetName().c_str(), "Dynamic cast opSrc to NodeProto failed.");
    return FAILED;
  }
  // onnx doesn't have transpose attr
  opDst.SetAttr("adj_x1", false);
  opDst.SetAttr("adj_x2", false);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("BatchMatMul")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::BatchMatMul",
                 "ai.onnx::9::BatchMatMul",
                 "ai.onnx::10::BatchMatMul",
                 "ai.onnx::11::BatchMatMul",
                 "ai.onnx::12::BatchMatMul",
                 "ai.onnx::13::BatchMatMul"})
  .ParseParamsFn(ParseParamsBatchMatMul)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
