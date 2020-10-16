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

namespace domi {
const std::string EXP_ATTR_BASE = "base";
const std::string EXP_ATTR_SCALE = "scale";
const std::string EXP_ATTR_SHIFT = "shift";
const float DEFAULT_BASE_VALUE = -1.0;
const float DEFAULT_SCALE_VALUE = 1.0;
const float DEFAULT_SHIFT_VALUE = 0.0;
Status ParserParamExp(const Message *op_src, ge::Operator &op)
{
    AutoMappingFn(op_src, op);
    op.SetAttr(EXP_ATTR_BASE, (float)DEFAULT_BASE_VALUE);
    op.SetAttr(EXP_ATTR_SCALE, (float)DEFAULT_SCALE_VALUE);
    op.SetAttr(EXP_ATTR_SHIFT, (float)DEFAULT_SHIFT_VALUE);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("Exp")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Exp")
    .ParseParamsFn(ParserParamExp)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
