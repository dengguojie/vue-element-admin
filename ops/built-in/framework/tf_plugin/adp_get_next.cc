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
 * \file adp_get_next.cc-
 * \brief
 */
#include "register/register.h"
#include "tensor.h"

#include "op_log.h"
namespace domi {
Status AutoMappingFnAdpGetNext(const Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["out"] = pair<string, string>("y", "output_types");
  if (AutoMappingFnDynamic(op_src, op, value) != SUCCESS) {
    OP_LOGE("AdpGetNext", "tensorflow plugin parser failed. auto mapping failed.");
    return FAILED;
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("AdpGetNext")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AdpGetNext")
    .ParseParamsFn(AutoMappingFnAdpGetNext)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
