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
#include <vector>
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;
Status ParseParamsLogSoftmax(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("ParseParamsLogSoftmax",
            "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int axis = -1;
  for (auto attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis = attr.i();
    }
  }
  std::vector<int> axes(1, axis);
  op_dest.SetAttr("axes", axes);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("LogSoftmaxV2")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::LogSoftmax")
  .OriginOpType({"ai.onnx::9::LogSoftmax",
                 "ai.onnx::12::LogSoftmax",
                 "ai.onnx::13::LogSoftmax"})
  .ParseParamsFn(ParseParamsLogSoftmax)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
