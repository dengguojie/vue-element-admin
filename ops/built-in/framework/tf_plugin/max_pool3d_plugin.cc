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
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "operator.h"
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
        OP_LOGE(op.GetName().c_str(), "update input x format failed.");
        return FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "update input x format success, now is %d", op.GetInputDesc(POS_0).GetFormat());

    ge::GeTensorDesc orgTensorY = op_dsc->GetOutputDesc(POS_0);
    orgTensorY.SetOriginFormat(ge::FORMAT_NDHWC);
    orgTensorY.SetFormat(ge::FORMAT_NDHWC);
    ret = op_dsc->UpdateOutputDesc(POS_0, orgTensorY);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update output y format failed.");
        return FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "update output y format success, now is %d", op.GetOutputDesc(POS_0).GetFormat());

    return SUCCESS;
}

REGISTER_CUSTOM_OP("MaxPool3D")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MaxPool3D")
    .ParseParamsFn(ParseParamsMaxPool3D)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
