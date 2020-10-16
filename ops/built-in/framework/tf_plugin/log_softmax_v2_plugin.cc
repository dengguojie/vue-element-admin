/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"

namespace domi {
Status ParseParamsLogSoftmaxMappingFn(const Message* op_src,
           ge::Operator& op) {
  AutoMappingFn(op_src, op);
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDesc orgTensorW = op_dsc->GetInputDesc(0);
  ge::GeTensorDesc orgTensorW1 = op_dsc->GetOutputDesc(0);
  orgTensorW.SetOriginFormat(ge::FORMAT_NHWC);
  orgTensorW1.SetOriginFormat(ge::FORMAT_NHWC);
  orgTensorW.SetFormat(ge::FORMAT_NHWC);
  orgTensorW1.SetFormat(ge::FORMAT_NHWC);
  auto ret = op_dsc->UpdateInputDesc(0, orgTensorW);
  auto ret1 = op_dsc->UpdateInputDesc(0, orgTensorW1);
  if(ret != ge::GRAPH_SUCCESS || ret1 != ge::GRAPH_SUCCESS) {
    return FAILED;
  }

  std::vector<int64_t> reduce_dims = {-1};
  if (op.GetAttr("axis", reduce_dims) != ge::GRAPH_SUCCESS) {
    OP_LOGW("LogSoftmaxV2", "GetAttr axis failed");
  }
  op.SetAttr("axes", reduce_dims);
  return SUCCESS;
}
REGISTER_CUSTOM_OP("LogSoftmaxV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LogSoftmax")
    .ParseParamsFn(ParseParamsLogSoftmaxMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi

