/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 *You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

namespace domi {
Status ParseParamsSoftmaxCrossEntropyWithLogitsMappingFn(const Message* op_src,
           ge::Operator& op) {
  AutoMappingFn(op_src, op);
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDesc orgTensorW = op_dsc->GetInputDesc(0);
  ge::GeTensorDesc orgTensorW1 = op_dsc->GetInputDesc(1);
  ge::GeTensorDesc orgTensorW2 = op_dsc->GetOutputDesc(0);
  ge::GeTensorDesc orgTensorW3 = op_dsc->GetOutputDesc(1);
  orgTensorW.SetOriginFormat(ge::FORMAT_NHWC);
  orgTensorW1.SetOriginFormat(ge::FORMAT_NHWC);
  orgTensorW2.SetOriginFormat(ge::FORMAT_NHWC);
  orgTensorW3.SetOriginFormat(ge::FORMAT_NHWC);
  orgTensorW.SetFormat(ge::FORMAT_NHWC);
  orgTensorW1.SetFormat(ge::FORMAT_NHWC);
  orgTensorW2.SetFormat(ge::FORMAT_NHWC);
  orgTensorW3.SetFormat(ge::FORMAT_NHWC);
  auto ret = op_dsc->UpdateInputDesc(0, orgTensorW);
  auto ret1 = op_dsc->UpdateInputDesc(1, orgTensorW1);
  auto ret2 = op_dsc->UpdateOutputDesc(0, orgTensorW2);
  auto ret3 = op_dsc->UpdateOutputDesc(1, orgTensorW3);
  if(ret != ge::GRAPH_SUCCESS || ret1 != ge::GRAPH_SUCCESS || ret2 !=
     ge::GRAPH_SUCCESS || ret3 != ge::GRAPH_SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}
REGISTER_CUSTOM_OP("SoftmaxCrossEntropyWithLogits")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SoftmaxCrossEntropyWithLogits")
    .ParseParamsFn(ParseParamsSoftmaxCrossEntropyWithLogitsMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
