// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
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
uint32_t kPosition = 3;

Status BasicLSTMCellParserParams(const std::vector<const google::protobuf::Message *> inside_nodes, ge::Operator &op) {
  OP_LOGI(op.GetName().c_str(), "Enter BasicLSTMCell fusion parser.");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "Get op desc failed.");
    return FAILED;
  }

  ge::GeTensorDesc input_desc = op_desc->GetInputDesc(kPosition);
  input_desc.SetOriginFormat(ge::FORMAT_HWCN);
  input_desc.SetFormat(ge::FORMAT_HWCN);

  if (op_desc->UpdateInputDesc(kPosition, input_desc) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update input desc fail, index:%u.", kPosition);
    return FAILED;
  }

  return SUCCESS;
}

Status ParseParamsBasicLSTMCell(const Message* op_src, ge::Operator& op) {

    AutoMappingFn(op_src, op);
    auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDesc orgTensorW = op_dsc->GetInputDesc(kPosition);
    orgTensorW.SetOriginFormat(ge::FORMAT_HWCN);
    orgTensorW.SetFormat(ge::FORMAT_HWCN);
    auto ret = op_dsc->UpdateInputDesc(kPosition, orgTensorW);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update filter format failed.");
        return FAILED;
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("BasicLSTMCell")
  .FrameworkType(TENSORFLOW)
  .OriginOpType("BasicLSTMCell")
  .ParseParamsFn(ParseParamsBasicLSTMCell)
  .FusionParseParamsFn(BasicLSTMCellParserParams)
  .ImplyType(ImplyType::TVM);
}
