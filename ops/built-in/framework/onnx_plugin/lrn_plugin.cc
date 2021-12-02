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
Status ParseParamsLRN(const Message *op_src, ge::Operator &op_dst) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dst.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  float alpha = 0.0001;
  float beta = 0.75;
  float bias = 1.0;
  int32_t size = -1;
  int32_t depth_radius = -1;
  bool bfind_size = false;

  for (auto attr : node->attribute()) {
    if (attr.name() == "alpha") {
      alpha = attr.f();
    } else if (attr.name() == "beta") {
      beta = attr.f();
    } else if (attr.name() == "bias") {
      bias = attr.f();
    } else if (attr.name() == "size") {
      bfind_size = true;
      size = attr.i();
    }
  }

  if (!bfind_size) {
    ONNX_PLUGIN_LOGE(op_dst.GetName().c_str(), "Message do not have attr size");
    return FAILED;
  }

  int even_base = 2;
  if (size % even_base == 0) {
    ONNX_PLUGIN_LOGE(op_dst.GetName().c_str(), "Attr size must be odd num");
    return FAILED;
  }
  alpha = alpha / size;
  depth_radius = (size - 1) / 2;
  op_dst.SetAttr("depth_radius", depth_radius);
  op_dst.SetAttr("bias", bias);
  op_dst.SetAttr("alpha", alpha);
  op_dst.SetAttr("beta", beta);
  op_dst.SetAttr("norm_region", "ACROSS_CHANNELS");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("LRN")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::LRN",
                 "ai.onnx::9::LRN",
                 "ai.onnx::10::LRN",
                 "ai.onnx::11::LRN",
                 "ai.onnx::12::LRN",
                 "ai.onnx::13::LRN"})
  .ParseParamsFn(ParseParamsLRN)
  .ImplyType(ImplyType::TVM);
}  //  namespace domi
