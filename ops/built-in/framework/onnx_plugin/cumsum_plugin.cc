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

namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsCumSum(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  bool exclusive = false;
  bool reverse = false;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "exclusive" &&
        attr.type() == ge::onnx::AttributeProto::INT) {
      exclusive = (attr.i() == 1);
    } else if (attr.name() == "reverse" &&
               attr.type() == ge::onnx::AttributeProto::INT) {
      reverse = (attr.i() == 1);
    }
  }
  op_dest.SetAttr("exclusive", exclusive);
  op_dest.SetAttr("reverse", reverse);

  return SUCCESS;
}

// register CumSum op info to GE
REGISTER_CUSTOM_OP("Cumsum")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::CumSum",
                 "ai.onnx::9::CumSum",
                 "ai.onnx::10::CumSum",
                 "ai.onnx::11::CumSum",
                 "ai.onnx::12::CumSum",
                 "ai.onnx::13::CumSum"})
  .ParseParamsFn(ParseParamsCumSum)
  .ImplyType(ImplyType::TVM);
}  // namespace domi