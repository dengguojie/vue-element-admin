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
 * \file hcom_broadcast_plugin.cpp
 * \brief
 */
#include "register/register.h"

#include "op_log.h"

namespace domi {
Status AutoMappingFnHcomBroadcast(const google::protobuf::Message* op_src, ge::Operator& op) {
  if (op_src == nullptr) {
    OP_LOGE("HcomBroadcast", "op[HcomBroadcast] tensorflow plugin parser[AutoMapping] failed. op_src is nullptr.");
    return FAILED;
  }
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("x", "T");
  value["out"] = pair<string, string>("y", "T");
  if (AutoMappingFnDynamic(op_src, op, value) == SUCCESS) {
    OP_LOGI("HcomBroadcast", "op[HcomBroadcast] tensorflow plugin parser[AutoMapping] success.");
    return SUCCESS;
  } else {
    OP_LOGE("HcomBroadcast", "op[HcomBroadcast] tensorflow plugin parser[AutoMapping] failed.");
    return FAILED;
  }
}

// register HcomBroadcast op to GE
REGISTER_CUSTOM_OP("HcomBroadcast")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("HcomBroadcast")
    .ParseParamsFn(AutoMappingFnHcomBroadcast)
    .ImplyType(ImplyType::HCCL);
}  // namespace domi
