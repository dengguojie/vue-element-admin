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
 * \file reduce_mean_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "op_log.h"

namespace domi {
Status AutoMappingFnReduceMean(const google::protobuf::Message* op_src, ge::Operator& op) {
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE("ReduceMean", "tensorflow plugin parser failed. auto mapping failed.");
    return FAILED;
  }
  op.SetAttr("noop_with_empty_axes", true);
  OP_LOGI("ReduceMean", "tensorflow plugin parser[AutoMapping] success.");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("ReduceMean")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Mean")
    .ParseParamsFn(AutoMappingFnReduceMean)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
