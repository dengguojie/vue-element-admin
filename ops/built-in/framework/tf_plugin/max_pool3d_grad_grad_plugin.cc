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
 * \file max_pool3d_grad_grad_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"

#include "op_log.h"

namespace domi {
// Replace ge ParseParams fuction to process graph maxpool3dgradgrad node attrs
Status ParseParamsMaxPool3DGradGrad(const Message* op_src, ge::Operator& op) {
  // Convert original tf graph maxpool3dgradgrad attrs to GE graph attrs
  AutoMappingFn(op_src, op);
  // Escape GE require attr [pads] check here
  std::vector<int32_t> padList = {0, 0, 0, 0, 0, 0};
  op.SetAttr("pads", padList);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("MaxPool3DGradGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MaxPool3DGradGrad")
    .ParseParamsFn(ParseParamsMaxPool3DGradGrad)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
