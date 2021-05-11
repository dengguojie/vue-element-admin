/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file caffe_axpy_v1_plugin.cpp
 * \brief
 */
#include "register/register.h"

using namespace ge;

namespace domi {
// transform caffe recognized data structure to ge recognized
Status ParseParamsAxpyV1(const ge::Operator &op_src, ge::Operator &op_dst)
{
    return SUCCESS;
}

REGISTER_CUSTOM_OP("AxpyV1")
    .FrameworkType(CAFFE)
    .OriginOpType("Axpy")
    .ParseParamsByOperatorFn(ParseParamsAxpyV1)
    .ImplyType(ImplyType::TVM);
}
