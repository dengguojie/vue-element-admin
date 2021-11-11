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
 * \file log_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "op_log.h"

namespace domi {
Status ParserParamLog(const Message* op_src, ge::Operator& op) {
  if (AutoMappingFn(op_src, op) != SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "AutoMappingFn failed.");
    return FAILED;
  }
  const std::string log_attr_base = "base";
  const float default_base_value = -1.0;
  op.SetAttr(log_attr_base, static_cast<float>(default_base_value));

  const std::string log_attr_scale = "scale";
  const float default_scale_value = 1.0;
  op.SetAttr(log_attr_scale, static_cast<float>(default_scale_value));

  const std::string log_attr_shift = "shift";
  const float default_shift_value = 0.0;
  op.SetAttr(log_attr_shift, static_cast<float>(default_shift_value));

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Log")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Log")
    .ParseParamsFn(ParserParamLog)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
