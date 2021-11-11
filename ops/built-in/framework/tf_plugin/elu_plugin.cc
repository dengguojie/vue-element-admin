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
 * \file elu_plugin.cpp
 * \brief
 */
#include "register/register.h"

namespace domi {
Status ParserParamElu(const Message* op_src, ge::Operator& op) {
  AutoMappingFn(op_src, op);
  const std::string elu_attr_alpha = "alpha";
  const float default_alpha_value = 1.0;
  op.SetAttr(elu_attr_alpha, static_cast<float>(default_alpha_value));
  return SUCCESS;
}
REGISTER_CUSTOM_OP("Elu")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Elu")
    .ParseParamsFn(ParserParamElu)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
