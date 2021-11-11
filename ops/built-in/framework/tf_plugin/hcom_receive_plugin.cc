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
 * \file hcom_receive_plugin.cpp
 * \brief
 */
#include "register/register.h"

#include "op_log.h"

namespace domi {
Status AutoMappingFnHcomReceive(const google::protobuf::Message* op_src, ge::Operator& op) {
  if (op_src == nullptr) {
    OP_LOGE("HcomReceive", "op[HcomReceive] tensorflow plugin parser[AutoMapping] failed. op_src is nullptr.");
    return FAILED;
  }
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE("HcomReceive", "tensorflow plugin parser failed. auto mapping failed.");
    return FAILED;
  }
  ge::DataType dataType;
  if (op.GetAttr("T", dataType) != ge::GRAPH_SUCCESS) {
    OP_LOGI("HcomReceive", "GetAttr T failed");
    return FAILED;
  }
  op.SetAttr("dtype", dataType);
  OP_LOGI("HcomReceive", "op[HcomReceive] tensorflow plugin parser[AutoMapping] success.");
  return SUCCESS;
}

// register HcomReceive op to GE
REGISTER_CUSTOM_OP("HcomReceive")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("HcomReceive")
    .ParseParamsFn(AutoMappingFnHcomReceive)
    .ImplyType(ImplyType::HCCL);
}  // namespace domi
