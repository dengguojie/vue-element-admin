/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file caffe_ascend_requant_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"

namespace domi {

Status ParseParamAscendRequant(const Message* op_src, ge::Operator& op_dest) {
  // set the default sqrt_mode and relu_flag value for DeQuant,
  // FE  will refresh the value.
  op_dest.SetAttr("relu_flag", false);

  return SUCCESS;
}

// register AscendDequant op info to GE
REGISTER_CUSTOM_OP("AscendRequant")
    .FrameworkType(CAFFE)
    .OriginOpType("Requant")
    .ParseParamsFn(ParseParamAscendRequant)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
