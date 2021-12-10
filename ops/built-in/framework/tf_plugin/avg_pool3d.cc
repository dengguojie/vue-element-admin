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
 * \file avg_pool3d.cpp
 * \brief
 */
#include "register/register.h"
#include "../../op_proto/util/axis_util.h"
#include "op_log.h"

namespace domi {
// Replace ge ParseParams fuction to process graph conv2d node attrs
Status ParseParamsAvgPool3D(const ge::Operator& op_src, ge::Operator& op) {
  ge::AscendString op_name;
  CHECK(op_src.GetName(op_name) != ge::GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return FAILED);

  // Convert original tf graph avg_pool3d attrs to GE graph attrs
  if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
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
    .ParseParamsByOperatorFn(ParseParamsAvgPool3D)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
