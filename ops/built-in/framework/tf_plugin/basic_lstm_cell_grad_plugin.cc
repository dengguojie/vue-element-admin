// Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
// This program is free software; you can redistribute it and/or modify
// it under the terms of the Apache License Version 2.0.You may not use
// this file except in compliance with the License.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// Apache License for more details at
// http:// www.apache.org/licenses/LICENSE-2.0


#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "operator.h"
#include "op_log.h"
namespace domi {
const int POS_0 = 0;
const int POS_1 = 1;

Status ParseParamsBasicLSTMCellInputGrad(const Message* op_src, ge::Operator& op) {
    AutoMappingFn(op_src, op);
    auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDesc orgTensorW = op_dsc->GetInputDesc(POS_1);
    orgTensorW.SetOriginFormat(ge::FORMAT_HWCN);
    orgTensorW.SetFormat(ge::FORMAT_HWCN);
    auto ret = op_dsc->UpdateInputDesc(POS_1, orgTensorW);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update filter format failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status ParseParamsBasicLSTMCellWeightGrad(const Message* op_src, ge::Operator& op) {
    AutoMappingFn(op_src, op);
    auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDesc orgTensorW = op_dsc->GetOutputDesc(POS_0);
    orgTensorW.SetOriginFormat(ge::FORMAT_HWCN);
    orgTensorW.SetFormat(ge::FORMAT_HWCN);
    auto ret = op_dsc->UpdateOutputDesc(POS_0, orgTensorW);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update filter format failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status ParseParamsBasicLSTMCellCStateGrad(const Message* op_src, ge::Operator& op) {
    AutoMappingFn(op_src, op);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("BasicLSTMCellCStateGrad")
  .FrameworkType(TENSORFLOW)
  .OriginOpType("BasicLSTMCellCStateGrad")
  .ParseParamsFn(ParseParamsBasicLSTMCellCStateGrad)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("BasicLSTMCellWeightGrad")
  .FrameworkType(TENSORFLOW)
  .OriginOpType("BasicLSTMCellWeightGrad")
  .ParseParamsFn(ParseParamsBasicLSTMCellWeightGrad)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("BasicLSTMCellInputGrad")
  .FrameworkType(TENSORFLOW)
  .OriginOpType("BasicLSTMCellInputGrad")
  .ParseParamsFn(ParseParamsBasicLSTMCellInputGrad)
  .ImplyType(ImplyType::TVM);
}
