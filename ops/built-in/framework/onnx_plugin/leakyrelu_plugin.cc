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
Status ParseParamsLeakyRelu(const Message *op_src, ge::Operator &op_dst) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dst.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  float negative_slope = 0.01f;
  for (auto attr : node->attribute()) {
    if (attr.name() == "alpha"&& attr.type() == ge::onnx::AttributeProto::FLOAT) {
      negative_slope = attr.f();
    }
  }
  op_dst.SetAttr("negative_slope", negative_slope);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("LeakyRelu")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::LeakyRelu",
                 "ai.onnx::9::LeakyRelu",
                 "ai.onnx::10::LeakyRelu",
                 "ai.onnx::11::LeakyRelu",
                 "ai.onnx::12::LeakyRelu",
                 "ai.onnx::13::LeakyRelu"})
  .ParseParamsFn(ParseParamsLeakyRelu)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
