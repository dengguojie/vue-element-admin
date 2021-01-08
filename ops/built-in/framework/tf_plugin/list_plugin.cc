/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file list_plugin.cc
 * \brief
 */
#include "register/register.h"

namespace domi {

// register EmptyTensorList op to GE
REGISTER_CUSTOM_OP("EmptyTensorList")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("EmptyTensorList")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListPushBack op to GE
REGISTER_CUSTOM_OP("TensorListPushBack")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListPushBack")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListPopBack op to GE
REGISTER_CUSTOM_OP("TensorListPopBack")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListPopBack")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListLength op to GE
REGISTER_CUSTOM_OP("TensorListLength")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListLength")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListElementShape op to GE
REGISTER_CUSTOM_OP("TensorListElementShape")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListElementShape")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListReserve op to GE
REGISTER_CUSTOM_OP("TensorListReserve")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListReserve")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListGetItem op to GE
REGISTER_CUSTOM_OP("TensorListGetItem")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListGetItem")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListSetItem op to GE
REGISTER_CUSTOM_OP("TensorListSetItem")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListSetItem")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

}  // namespace domi
