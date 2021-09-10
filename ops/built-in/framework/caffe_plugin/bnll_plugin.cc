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
 * \file bnll_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {

Status ParseParamsBNLL(const Message* op_origin, ge::Operator& op_dest) {
  // trans op_src to op_dest
  if (op_origin == nullptr) {
    OP_LOGE("op_origin get failed.\n");
    return FAILED;
  }
  
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);
  if (layer == nullptr) {
    OP_LOGE("Dynamic cast op_src to LayerParameter failed.\n");
    return FAILED;
  }

  return SUCCESS;
}

// register BNLL op info to GE
REGISTER_CUSTOM_OP("BNLL")
    .FrameworkType(CAFFE)
    .OriginOpType("BNLL")
    .ParseParamsFn(ParseParamsBNLL)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
