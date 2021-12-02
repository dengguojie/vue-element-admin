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
 * \file tensor_array_plugin.cpp
 * \brief
 */
#include "register/register.h"

#include "op_log.h"

namespace domi {
Status AutoMappingFnTensorArray(const Message* op_src, ge::Operator& op) {
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE("TensorArray", "Tensorflow plugin parser failed. Auto mapping failed.");
    return FAILED;
  }
  ge::Operator::OpListInt elem_dims;
  if (op.GetAttr("element_shape", elem_dims) != ge::GRAPH_SUCCESS) {
    OP_LOGE("TensorArray", "GetAttr element_shape failed");
    return FAILED;
  }
  if (elem_dims == ge::UNKNOWN_SHAPE) {
    op.SetAttr("element_shape", ge::UNKNOWN_RANK);
  }
  OP_LOGI("TensorArray", "Op[TensorArray] tensorflow plugin parser[AutoMapping] success.");
  return SUCCESS;
}

// register TensorArray op to GE
REGISTER_CUSTOM_OP("TensorArray")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArrayV3")
    .ParseParamsFn(AutoMappingFnTensorArray)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
