/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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

// register TensorListPushBackBatch op to GE
REGISTER_CUSTOM_OP("TensorListPushBackBatch")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListPushBackBatch")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListStack op to GE
REGISTER_CUSTOM_OP("TensorListStack")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListStack")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListConcatV2 op to GE
REGISTER_CUSTOM_OP("TensorListConcatV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListConcatV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListSplit op to GE
REGISTER_CUSTOM_OP("TensorListSplit")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListSplit")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListFromTensor op to GE
REGISTER_CUSTOM_OP("TensorListFromTensor")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListFromTensor")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListResize op to GE
REGISTER_CUSTOM_OP("TensorListResize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListResize")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListGather op to GE
REGISTER_CUSTOM_OP("TensorListGather")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListGather")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListScatterV2 op to GE
REGISTER_CUSTOM_OP("TensorListScatterV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListScatterV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListScatterIntoExistingList op to GE
REGISTER_CUSTOM_OP("TensorListScatterIntoExistingList")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListScatterIntoExistingList")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorListConcatLists op to GE
REGISTER_CUSTOM_OP("TensorListConcatLists")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorListConcatLists")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
