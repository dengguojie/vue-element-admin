/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file caffe_ascend_dequant_s16_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"

namespace domi {
Status ParseParamAscendDequantS16(const Message* op_src, ge::Operator& op_dest) {
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
