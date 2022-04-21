/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file nonzero_with_value_plugin.cc
 * \brief
 */
#include "register/register.h"
#include "op_log.h"

namespace domi {
Status AutoMappingFnNonZeroWithValue(const google::protobuf::Message* op_src, ge::Operator& op) {
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE("NonZeroWithValue", "Tensorflow plugin parser failed. Auto mapping failed.");
    return FAILED;
  }
  ge::DataType outputDataType;
  op.GetAttr("output_type", outputDataType);
  (void)op.SetAttr("dtype", outputDataType);
  return ret;
}

REGISTER_CUSTOM_OP("NonZeroWithValue")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NonZeroWithValue")
    .ParseParamsFn(AutoMappingFnNonZeroWithValue)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
