/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file cross_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"

namespace domi {
Status AutoMappingFnCross(const google::protobuf::Message* op_src, ge::Operator& op) {
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE("Cross", "tensorflow plugin parser failed. auto mapping failed.");
    return FAILED;
  }
  int32_t dim = -1;
  op.SetAttr("dim", dim);
  OP_LOGI("Cross", "op[Cross] tensorflow plugin parser[AutoMapping] success.");
  return SUCCESS;
}
REGISTER_CUSTOM_OP("Cross")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Cross")
    .ParseParamsFn(AutoMappingFnCross)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
