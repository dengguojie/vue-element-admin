/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
 * \file ascend_dequant_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"
#include "op_log.h"

namespace domi {
Status AutoMappingFnDequant(const google::protobuf::Message* op_src, ge::Operator& op) {
  if (AutoMappingFn(op_src, op) != SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "auto mapping failed.");
    return FAILED;
  }
  (void)op.SetAttr("tf_tag", "tf");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("AscendDequant")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AscendDequant")
    .ParseParamsFn(AutoMappingFnDequant)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
