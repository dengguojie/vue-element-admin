/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2018. All rights reserved.
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
 * \file assert_plugin.cpp
 * \brief
 */
#include "register/register.h"

namespace domi {
Status AutoMappingFnAssert(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("input_data", "T");
  return AutoMappingFnDynamic(op_src, op, value);
}

// register Assert op to GE
REGISTER_CUSTOM_OP("Assert")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Assert")
    .ParseParamsFn(AutoMappingFnAssert)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
