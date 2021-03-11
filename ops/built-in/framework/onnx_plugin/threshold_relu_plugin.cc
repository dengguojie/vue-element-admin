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
#include <string>

#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsThresholdedRelu(const Message* op_src,
                                  ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("ParseParamsThresholdedRelu",
            "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  float alpha = 1.0;
  for (auto attr : node->attribute()) {
    if (attr.name() == "alpha") {
      alpha = attr.f();
    }
  }
  op_dest.SetAttr("alpha", alpha);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("ThresholdedRelu")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::9::ThresholdedRelu",
                 "ai.onnx::10::ThresholdedRelu",
                 "ai.onnx::11::ThresholdedRelu",
                 "ai.onnx::12::ThresholdedRelu"})
  .ParseParamsFn(ParseParamsThresholdedRelu)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
