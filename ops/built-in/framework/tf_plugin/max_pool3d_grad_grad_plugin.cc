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

// Replace ge ParseParams fuction to process graph maxpool3dgradgrad node attrs
Status ParseParamsMaxPool3DGradGrad(const Message* op_src, ge::Operator& op) {
    // Convert original tf graph maxpool3dgradgrad attrs to GE graph attrs
    AutoMappingFn(op_src, op);
    // Escape GE require attr [pads] check here
    std::vector<int32_t> padList = {0,0,0,0,0,0};
    op.SetAttr("pads", padList);
    return SUCCESS;
}


REGISTER_CUSTOM_OP("MaxPool3DGradGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MaxPool3DGradGrad")
    .ParseParamsFn(ParseParamsMaxPool3DGradGrad)
    .ImplyType(ImplyType::TVM);
}  // namespace domi

