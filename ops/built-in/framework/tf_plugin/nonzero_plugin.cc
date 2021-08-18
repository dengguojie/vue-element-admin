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
 * \file nonzero_plugin.cc
 * \brief
 */
#include "register/register.h"
#include "op_log.h"

namespace domi {
Status AutoMappingFnNonZero(const google::protobuf::Message* op_src, ge::Operator& op) {
  Status ret = AutoMappingFn(op_src, op);
  ge::DataType dataType;
  op.GetAttr("output_type", dataType);
  op.SetAttr("dtype", dataType);
  return ret;
}

REGISTER_CUSTOM_OP("NonZero")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NonZero")
    .ParseParamsFn(AutoMappingFnNonZero)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
