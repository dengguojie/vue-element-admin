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
 * \file basic_lstm_cell_grad_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"

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
    OP_LOGE(TbeGetName(op).c_str(), "update filter format failed.");
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
    OP_LOGE(TbeGetName(op).c_str(), "update filter format failed.");
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
}  // namespace domi
