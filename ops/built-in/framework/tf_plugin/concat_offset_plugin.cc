/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
 * \file concat_offset_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "op_log.h"

namespace domi {
Status AutoMappingFnConcatOffsetD(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("x", "N");
  value["out"] = pair<string, string>("y", "N");
  OP_CHECK(AutoMappingFnDynamic(op_src, op, value) != SUCCESS,
           OP_LOGE(TbeGetName(op), "AutoMappingFn failed."), return FAILED);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("ConcatOffset")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ConcatOffset")
    .ParseParamsFn(AutoMappingFnConcatOffsetD)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
