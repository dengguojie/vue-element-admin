/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2018. All rights reserved.
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
 * \file getspan_plugin.cpp
 * \brief
 */
#include "register/register.h"

namespace domi {
// test_reduction is the type name of the operator in the OM model.
// It can be specified randomly and cannot be the same as an existing type name.
// It is case sensitive.
REGISTER_CUSTOM_OP("GetSpan")
    .FrameworkType(TENSORFLOW)     // Enumerated type. The options are as follows:
                                   // CAFFE, TENSORFLOW
    .OriginOpType("GetSpan")       // // Reduction indicates the type name of the
                                   // operator in the caffe framework.
    .ParseParamsFn(AutoMappingFn)  // AutoMappingFn indicates automatic mapping
                                   // the parameters of op.
    .ImplyType(ImplyType::TVM);    // Implementation type. Enumerated type, The
                                   // options are as follows: TVM, AI_CPU.
}  // namespace domi
