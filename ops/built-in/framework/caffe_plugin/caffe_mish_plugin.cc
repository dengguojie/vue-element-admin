/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All 
rights reserved.
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

#include "proto/caffe/caffe.pb.h"
#include "register/register.h"

namespace domi {

Status ParseParamMish(const Message *op_src, ge::Operator &op_dest) {
  return SUCCESS;
}

// register Mish op info to GE
REGISTER_CUSTOM_OP("Mish")
  .FrameworkType(CAFFE)
  .OriginOpType("Mish")
  .ParseParamsFn(ParseParamMish)
  .ImplyType(ImplyType::TVM);
}  // namespace domi






