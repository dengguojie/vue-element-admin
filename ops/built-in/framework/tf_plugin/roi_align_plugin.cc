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
 * \file roi_align_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

#define NET_2D_H1_ROI_ALIGN_POOL_HEIGHT 7
#define NET_2D_H1_ROI_ALIGN_POOL_WIDTH 7

namespace domi {
Status ROIAlignParams(const std::vector<const google::protobuf::Message*> insideNodes, ge::Operator& op) {
  OP_LOGI(TbeGetName(op).c_str(), "Enter ROIAlign fusion parser.");
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (opDesc == nullptr) {
    OP_LOGE(TbeGetName(op).c_str(), "Get op desc failed.");
    return FAILED;
  }
  if (!ge::AttrUtils::SetFloat(opDesc, "spatial_scale", 1.0)) {
    OP_LOGE(TbeGetName(op).c_str(), "Set spatial_scale failed.");
    return FAILED;
  }
  if (!ge::AttrUtils::SetInt(opDesc, "pooled_height", NET_2D_H1_ROI_ALIGN_POOL_HEIGHT)) {
    OP_LOGE(TbeGetName(op).c_str(), "Set pooled_height failed.");
    return FAILED;
  }
  if (!ge::AttrUtils::SetInt(opDesc, "pooled_width", NET_2D_H1_ROI_ALIGN_POOL_WIDTH)) {
    OP_LOGE(TbeGetName(op).c_str(), "Set pooled_width failed.");
    return FAILED;
  }
  return SUCCESS;
}

REGISTER_CUSTOM_OP("ROIAlign")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ROIAlign")
    .FusionParseParamsFn(ROIAlignParams)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
