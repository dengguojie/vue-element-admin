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
Status ParseParamsSize(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("Size", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  // set output's default type to int64.
  std::string output_type = "int64";
  op_dest.SetAttr("dtype", output_type);
  return SUCCESS;
}
// register Size op info to GE
REGISTER_CUSTOM_OP("Size")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::9::Size",
                 "ai.onnx::11::Size",
                 "ai.onnx::12::Size",
                 "ai.onnx::13::Size"})
  .ParseParamsFn(ParseParamsSize)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
