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

#include "proto/caffe/caffe.pb.h"
#include "register/register.h"

namespace domi {

Status ParseParamAscendDequantS16(const Message *op_src, ge::Operator &op_dest) {
  // set the default sqrt_mode and relu_flag value for DeQuant,
  // FE  will refresh the value.
  op_dest.SetAttr("relu_flag", false);

  return SUCCESS;
}

// register AscendDequant op info to GE
REGISTER_CUSTOM_OP("AscendDequantS16")
  .FrameworkType(CAFFE)
  .OriginOpType("DequantS16")
  .ParseParamsFn(ParseParamAscendDequantS16)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
