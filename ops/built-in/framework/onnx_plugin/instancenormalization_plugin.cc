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
static const float EPSILON_DEFAULT = 1e-05;

Status ParseParamsInstanceNormalization(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  float epsilon = EPSILON_DEFAULT;
  
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "epsilon" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
        epsilon = attr.f();
  }
  }
  op_dest.SetAttr("epsilon", epsilon);
  op_dest.SetAttr("data_format", "NCHW");
  
  return SUCCESS;
}
// register Add op info to GE
REGISTER_CUSTOM_OP("InstanceNorm")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::InstanceNormalization",
                   "ai.onnx::9::InstanceNormalization",
                   "ai.onnx::10::InstanceNormalization",
                   "ai.onnx::11::InstanceNormalization",
                   "ai.onnx::12::InstanceNormalization",
                   "ai.onnx::13::InstanceNormalization"})
    .ParseParamsFn(ParseParamsInstanceNormalization)
    .ImplyType(ImplyType::TVM);
}  // namespace domi