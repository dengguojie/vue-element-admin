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
Status ParseParamsLeakyRelu(const Message *op_src, ge::Operator &op_dst) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("Floor", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  float negative_slope = 0.0f;
  bool bfind_alpha = false;
  for (auto attr : node->attribute()) {
    if (attr.name() == "alpha") {
      bfind_alpha = true;
      negative_slope = attr.f();
      break;
    }
  }

  if (!bfind_alpha) {
    negative_slope = 0.01f;
  }
  op_dst.SetAttr("negative_slope", negative_slope);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("LeakyRelu")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::LeakyRelu")
  .ParseParamsFn(ParseParamsLeakyRelu)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
