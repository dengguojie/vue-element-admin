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
 * \file exp_plugin.cpp
 * \brief
 */
#include "register/register.h"

namespace domi {
Status ParserParamExp(const Message* op_src, ge::Operator& op) {
  AutoMappingFn(op_src, op);

  const std::string exp_attr_base = "base";
  const float default_base_value = -1.0;
  op.SetAttr(exp_attr_base, static_cast<float>(default_base_value));

  const std::string exp_attr_scale = "scale";
  const float default_scale_value = 1.0;
  op.SetAttr(exp_attr_scale, static_cast<float>(default_scale_value));

  const std::string exp_attr_shift = "shift";
  const float default_shift_value = 0.0;
  op.SetAttr(exp_attr_shift, static_cast<float>(default_shift_value));
  return SUCCESS;
}

REGISTER_CUSTOM_OP("Exp")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Exp")
    .ParseParamsFn(ParserParamExp)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
