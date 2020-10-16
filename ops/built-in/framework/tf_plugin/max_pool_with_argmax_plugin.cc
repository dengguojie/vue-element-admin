/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "graph/utils/op_desc_utils.h"
#include "register/register.h"

namespace domi {
Status MaxPoolWithArgMappingFn(const Message* op_src, ge::Operator& op) {
  AutoMappingFn(op_src, op);
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDesc tensor_descw = op_dsc->GetInputDesc(0);
  ge::GeTensorDesc tensor_descw1 = op_dsc->GetOutputDesc(0);
  ge::GeTensorDesc tensor_descw2 = op_dsc->GetOutputDesc(1);
  tensor_descw.SetOriginFormat(ge::FORMAT_NHWC);
  tensor_descw1.SetOriginFormat(ge::FORMAT_NHWC);
  tensor_descw2.SetOriginFormat(ge::FORMAT_NHWC);
  tensor_descw.SetFormat(ge::FORMAT_NHWC);
  tensor_descw1.SetFormat(ge::FORMAT_NHWC);
  tensor_descw2.SetFormat(ge::FORMAT_NHWC);
  auto ret = op_dsc->UpdateInputDesc(0, tensor_descw);
  auto ret1 = op_dsc->UpdateOutputDesc(0, tensor_descw1);
  auto ret2 = op_dsc->UpdateOutputDesc(1, tensor_descw2);
  if (ret != ge::GRAPH_SUCCESS || ret1 != ge::GRAPH_SUCCESS ||
      ret2 != ge::GRAPH_SUCCESS ) {
    return FAILED;
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("MaxPoolWithArgmax")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MaxPoolWithArgmax")
    .ParseParamsFn(MaxPoolWithArgMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
