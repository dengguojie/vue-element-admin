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
 * \file ocr_plugin.cpp
 * \brief
 */
#include "register/register.h"

namespace domi {
REGISTER_CUSTOM_OP("BatchEnqueue")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BatchEnqueue")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

REGISTER_CUSTOM_OP("OCRRecognitionPreHandle")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OCRRecognitionPreHandle")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

REGISTER_CUSTOM_OP("OCRDetectionPreHandle")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OCRDetectionPreHandle")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

REGISTER_CUSTOM_OP("OCRIdentifyPreHandle")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OCRIdentifyPreHandle")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

REGISTER_CUSTOM_OP("BatchDilatePolys")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BatchDilatePolys")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

REGISTER_CUSTOM_OP("OCRFindContours")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OCRFindContours")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

REGISTER_CUSTOM_OP("Dequeue")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Dequeue")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

REGISTER_CUSTOM_OP("OCRDetectionPostHandle")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OCRDetectionPostHandle")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

REGISTER_CUSTOM_OP("ResizeAndClipPolys")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResizeAndClipPolys")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
