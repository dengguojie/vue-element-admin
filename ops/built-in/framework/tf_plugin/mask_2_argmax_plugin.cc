/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
 * \file mask_2_argmax_plugin.cpp
 * \brief
 */
#include "graph/utils/op_desc_utils.h"
#include "register/register.h"

namespace domi {
REGISTER_CUSTOM_OP("Mask2Argmax")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Mask2Argmax")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
