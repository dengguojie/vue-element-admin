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
 * \file max_pool3d_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"

#include "op_log.h"

namespace domi {
const int POS_0 = 0;

Status ParseParamsMaxPool3D(const Message* op_src, ge::Operator& op) {
  AutoMappingFn(op_src, op);

  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);

  ge::GeTensorDesc orgTensorX = op_dsc->GetInputDesc(POS_0);
  orgTensorX.SetOriginFormat(ge::FORMAT_NDHWC);
  orgTensorX.SetFormat(ge::FORMAT_NDHWC);
  auto ret = op_dsc->UpdateInputDesc(POS_0, orgTensorX);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "update input x format failed.");
    return FAILED;
  }
  OP_LOGI(TbeGetName(op).c_str(), "update input x format success, now is %d", op.GetInputDesc(POS_0).GetFormat());

  ge::GeTensorDesc orgTensorY = op_dsc->GetOutputDesc(POS_0);
  orgTensorY.SetOriginFormat(ge::FORMAT_NDHWC);
  orgTensorY.SetFormat(ge::FORMAT_NDHWC);
  ret = op_dsc->UpdateOutputDesc(POS_0, orgTensorY);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "update output y format failed.");
    return FAILED;
  }
  OP_LOGI(TbeGetName(op).c_str(), "update output y format success, now is %d", op.GetOutputDesc(POS_0).GetFormat());

  std::vector<int32_t> padList = {0, 0, 0, 0, 0, 0};
  op.SetAttr("pads", padList);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("MaxPool3D")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MaxPool3D")
    .ParseParamsFn(ParseParamsMaxPool3D)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
