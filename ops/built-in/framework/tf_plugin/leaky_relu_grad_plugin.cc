/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file leaky_relu_grad_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {

Status AutoMappingFnLeakyReluGrad(const google::protobuf::Message* op_src, ge::Operator& op) {
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE("LeakyReluGrad", "tensorflow plugin parser failed. auto mapping failed.");
    return FAILED;
  }
  float alpha;
  if (op.GetAttr("alpha", alpha) != ge::GRAPH_SUCCESS) {
    OP_LOGI("LeakyReluGrad", "GetAttr alpha failed");
    return FAILED;
  }
  op.SetAttr("negative_slope", alpha);
  OP_LOGI("LeakyReluGrad", "op[LeakyReluGrad] tensorflow plugin parser success.");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("LeakyReluGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LeakyReluGrad")
    .ParseParamsFn(AutoMappingFnLeakyReluGrad)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
