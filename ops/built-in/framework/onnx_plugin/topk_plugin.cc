/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
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

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "operator.h"
#include "op_log.h"
#include "graph/utils/op_desc_utils.h"

namespace domi {
Status ParseParamsTopK(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("TopK", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int dim = -1;
  bool sorted = true;
  bool largest = true;
  for (const auto& attr: node->attribute()) {
    if (attr.name() == "sorted" && attr.type() == ge::onnx::AttributeProto::INT) {
      sorted = static_cast<bool>(attr.i());
    } else if (attr.name() == "largest" && attr.type() == ge::onnx::AttributeProto::INT) {
      largest = static_cast<bool>(attr.i());
    } else if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      dim = attr.i();
    }
  }

  op_dest.SetAttr("sorted", sorted);
  op_dest.SetAttr("dim", dim);
  op_dest.SetAttr("largest", largest);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("TopK")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::TopK",
                   "ai.onnx::10::TopK",
                   "ai.onnx::11::TopK",
                   "ai.onnx::12::TopK"})
    .ParseParamsFn(ParseParamsTopK)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
