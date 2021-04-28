/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file einsum_plugin.cc
 * \brief
 */
#include "register/register.h"
#include "op_log.h"

namespace domi {
Status AutoMappingFnEinSum(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("x", "N");
  AutoMappingFnDynamic(op_src, op, value);

  int num = 1;
  if (op.GetAttr("N", num) != ge::GRAPH_SUCCESS) {
    OP_LOGE("EinSum", "op[EinSum] GetAttr N failed.");
    return FAILED;
  }
  op.SetAttr("tensor_size", num);
  OP_LOGI("EinSum", "op[EinSum] tensorflow plugin parser[AutoMappingFnDynamic] success.");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("EinSum")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Einsum")
    .ParseParamsFn(AutoMappingFnEinSum)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
