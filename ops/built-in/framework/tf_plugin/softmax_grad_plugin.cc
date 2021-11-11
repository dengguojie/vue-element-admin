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
 * \file softmax_grad_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

namespace domi {
Status ParseParamsSoftmaxGradMappingFn(const Message* op_src, ge::Operator& op) {
  AutoMappingFn(op_src, op);
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDesc orgTensorW = op_dsc->GetInputDesc(0);
  ge::GeTensorDesc orgTensorW1 = op_dsc->GetInputDesc(1);
  ge::GeTensorDesc orgTensorW2 = op_dsc->GetOutputDesc(0);
  orgTensorW.SetOriginFormat(ge::FORMAT_NHWC);
  orgTensorW1.SetOriginFormat(ge::FORMAT_NHWC);
  orgTensorW2.SetOriginFormat(ge::FORMAT_NHWC);
  orgTensorW.SetFormat(ge::FORMAT_NHWC);
  orgTensorW1.SetFormat(ge::FORMAT_NHWC);
  orgTensorW2.SetFormat(ge::FORMAT_NHWC);
  auto ret = op_dsc->UpdateInputDesc(0, orgTensorW);
  auto ret1 = op_dsc->UpdateInputDesc(1, orgTensorW1);
  auto ret2 = op_dsc->UpdateOutputDesc(0, orgTensorW2);
  if (ret != ge::GRAPH_SUCCESS || ret1 != ge::GRAPH_SUCCESS || ret2 != ge::GRAPH_SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}
REGISTER_CUSTOM_OP("SoftmaxGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SoftmaxGrad")
    .ParseParamsFn(ParseParamsSoftmaxGradMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
