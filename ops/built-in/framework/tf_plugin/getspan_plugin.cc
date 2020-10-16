/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this
 * file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "register/register.h"

namespace domi {
// test_reduction is the type name of the operator in the OM model.
// It can be specified randomly and cannot be the same as an existing type name.
// It is case sensitive.
REGISTER_CUSTOM_OP("GetSpan")
    .FrameworkType(TENSORFLOW)  // Enumerated type. The options are as follows:
                                // CAFFE, TENSORFLOW
    .OriginOpType("GetSpan")    // // Reduction indicates the type name of the
                                // operator in the caffe framework.
    .ParseParamsFn(AutoMappingFn)  // AutoMappingFn indicates automatic mapping
                                   // the parameters of op.
    .ImplyType(ImplyType::TVM);    // Implementation type. Enumerated type, The
                                   // options are as follows: TVM, AI_CPU.
}  // namespace domi
