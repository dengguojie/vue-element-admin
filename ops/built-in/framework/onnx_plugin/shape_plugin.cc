/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
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
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include <string>
#include <vector>

namespace domi {

Status ParseParamsShape(const Message *op_src, ge::Operator &op_dest) {
  op_dest.SetAttr("dtype", static_cast<uint32_t>(ge::DT_INT64));
  return SUCCESS;
}

// register Add op info to GE
REGISTER_CUSTOM_OP("Shape")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::Shape")
  .ParseParamsFn(ParseParamsShape)
  .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
