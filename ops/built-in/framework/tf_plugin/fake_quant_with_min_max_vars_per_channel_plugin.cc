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
 * \file fake_quant_with_min_max_vars_per_channel_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"

namespace domi {
Status FakeQuantWithMinMaxVarsPerChannelMappingFn(const Message* op_src, ge::Operator& op) {
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE("FakeQuantWithMinMaxVarsPerChannel", "tensorflow plugin parser failed. auto mapping failed.");
    return FAILED;
  }
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDesc org_tensor_w = op_dsc->GetInputDesc(0);
  ge::GeTensorDesc org_tensor_w1 = op_dsc->GetOutputDesc(0);
  org_tensor_w.SetOriginFormat(ge::FORMAT_NHWC);
  org_tensor_w1.SetOriginFormat(ge::FORMAT_NHWC);
  org_tensor_w.SetFormat(ge::FORMAT_NHWC);
  org_tensor_w1.SetFormat(ge::FORMAT_NHWC);
  ret = op_dsc->UpdateInputDesc(0, org_tensor_w);
  auto ret1 = op_dsc->UpdateOutputDesc(0, org_tensor_w1);
  if (ret != ge::GRAPH_SUCCESS || ret1 != ge::GRAPH_SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

REGISTER_CUSTOM_OP("FakeQuantWithMinMaxVarsPerChannel")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FakeQuantWithMinMaxVarsPerChannel")
    .ParseParamsFn(FakeQuantWithMinMaxVarsPerChannelMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
