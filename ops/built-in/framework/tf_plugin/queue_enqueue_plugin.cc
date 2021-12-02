/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file queue_enqueue_plugin.cpp
 * \brief
 */
#include "register/register.h"

namespace domi {
Status AutoMappingFnEnqueue(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("components", "Tcomponents");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

// register QueueEnqueue op to GE
REGISTER_CUSTOM_OP("QueueEnqueue")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("QueueEnqueueV2")
    .ParseParamsFn(AutoMappingFnEnqueue)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
