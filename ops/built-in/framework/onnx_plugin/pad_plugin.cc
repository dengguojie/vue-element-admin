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
Status ParseParamsPad(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    OP_LOGE("Pad", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  op_dest.SetAttr("paddings_contiguous", false);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("PadV3")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::Pad")
    .ParseParamsFn(ParseParamsPad)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
