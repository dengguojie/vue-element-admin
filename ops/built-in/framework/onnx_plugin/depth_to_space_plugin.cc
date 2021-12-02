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

namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsDepthToSpace(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int block_size = 0;
  std::string mode = "DCR";
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "blocksize" && attr.type() == ge::onnx::AttributeProto::INT) {
      block_size = attr.i();
    } else if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      mode = attr.s();
    }
  }

  if (block_size < 2) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "blocksize, specifying the size of the spatial block, should be >=2, ");
    return FAILED;
  }
  
  if (ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, true) != SUCCESS) {
    return FAILED;
  }
  
  op_dest.SetAttr("block_size", block_size);
  op_dest.SetAttr("mode", mode);
  // set attr data format to NCHW.
  std::string output_format = "NCHW";
  op_dest.SetAttr("data_format", output_format);
  return SUCCESS;
}
// register DepthToSpace op info to GE
REGISTER_CUSTOM_OP("DepthToSpace")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::DepthToSpace",
                   "ai.onnx::9::DepthToSpace",
                   "ai.onnx::10::DepthToSpace",
                   "ai.onnx::11::DepthToSpace",
                   "ai.onnx::12::DepthToSpace",
                   "ai.onnx::13::DepthToSpace"})
    .ParseParamsFn(ParseParamsDepthToSpace)
    .ImplyType(ImplyType::TVM);
}  // namespace domi