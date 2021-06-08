/* Copyright (C) 2020-2021. Huawei Technologies Co., Ltd. All
rights reserved.
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
#include "graph/operator.h"

using namespace ge;

namespace domi
 {
// Caffe ParseParams
Status ParseParamConv2D(const Operator& op_src, ge::Operator& op_dest)
{
    // To do: Implement the operator plug-in by referring to the TBE Operator Development Guide.
    return SUCCESS;
}

// register op info to GE
REGISTER_CUSTOM_OP("Conv2D")
    .FrameworkType(CAFFE)    // type: CAFFE, TENSORFLOW
    .OriginOpType("Conv2D")       // name in caffe module
    .ParseParamsByOperatorFn(ParseParamConv2D);
} // namespace domi
