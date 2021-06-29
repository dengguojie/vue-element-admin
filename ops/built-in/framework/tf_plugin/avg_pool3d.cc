/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file avg_pool3d.cpp
 * \brief
 */
#include "register/register.h"
#include "op_log.h"

namespace domi {

// Replace ge ParseParams fuction to process graph conv2d node attrs
Status ParseParamsAvgPool3D(const Message* op_src, ge::Operator& op) {
  // Convert original tf graph avg_pool3d attrs to GE graph attrs
  if (AutoMappingFn(op_src, op) != SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "AutoMappingFn failed.");
    return FAILED;
  }

  // Escape GE require attr [pads] check here
  std::vector<int32_t> padList = {0, 0, 0, 0, 0, 0};
  op.SetAttr("pads", padList);
  op.SetAttr("count_include_pad", false);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("AvgPool3D")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AvgPool3D")
    .ParseParamsFn(ParseParamsAvgPool3D)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
