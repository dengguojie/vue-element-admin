/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: YoloV3 postprocessor tf plugin cpp file
 * Author:
 * Create: 2020-6-11
 * Note:
 */

#include "register/register.h"

namespace domi {
// register tik op info to GE
REGISTER_CUSTOM_OP("Yolov3PostProcessor")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Yolov3PostProcessor")
    .ParseParamsFn(AutoMappingFn);
}
