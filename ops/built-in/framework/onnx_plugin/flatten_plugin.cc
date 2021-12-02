/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Create: 2020-08-27
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

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
using NodeProto = ge::onnx::NodeProto;

Status ParseParamsFlatten(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int axis = 1;
  for (auto attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis = attr.i();
    }
  }
  op_dest.SetAttr("axis", axis);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Flatten")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::Flatten",
                 "ai.onnx::9::Flatten",
                 "ai.onnx::10::Flatten",
                 "ai.onnx::11::Flatten",
                 "ai.onnx::12::Flatten",
                 "ai.onnx::13::Flatten"})
  .ParseParamsFn(ParseParamsFlatten)
  .ImplyType(ImplyType::TVM);
}  // namespace domi