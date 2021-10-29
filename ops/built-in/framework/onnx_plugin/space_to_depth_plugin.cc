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
Status ParseParamsSpaceToDepth(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int blocksize_val = 2;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "blocksize" && attr.type() == ge::onnx::AttributeProto::INT) {
      blocksize_val = attr.i();
      break;
    }
  }
  
  if (ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, false) != SUCCESS) {
    return FAILED;
  }
  if (ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, true) != SUCCESS) {
    return FAILED;
  }

  op_dest.SetAttr("block_size", blocksize_val);
  op_dest.SetAttr("data_format", "NCHW");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("SpaceToDepth")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::SpaceToDepth",
                 "ai.onnx::9::SpaceToDepth",
                 "ai.onnx::10::SpaceToDepth",
                 "ai.onnx::11::SpaceToDepth",
                 "ai.onnx::12::SpaceToDepth",
                 "ai.onnx::13::SpaceToDepth"})
  .ParseParamsFn(ParseParamsSpaceToDepth)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
