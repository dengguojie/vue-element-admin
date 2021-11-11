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
 * \file lru_cache_plugin.cpp
 * \brief
 */
#include "register/register.h"

namespace domi {
// register LruCache op to GE
REGISTER_CUSTOM_OP("LruCache")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LruCache")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register CacheAdd op to GE
REGISTER_CUSTOM_OP("CacheAdd")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CacheAdd")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register CacheRemoteIndexToLocal op to GE
REGISTER_CUSTOM_OP("CacheRemoteIndexToLocal")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CacheRemoteIndexToLocal")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register CacheAllIndexToLocal op to GE
REGISTER_CUSTOM_OP("CacheAllIndexToLocal")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CacheAllIndexToLocal")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);
}
