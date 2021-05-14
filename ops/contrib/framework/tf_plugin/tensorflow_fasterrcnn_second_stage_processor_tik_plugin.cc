/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Plugin for Operator faster rcnn second stage processor
 * Author: 
 * Create: 2020-06-17
 */

#include "register/register.h"

namespace domi {
    // register FasterrcnnSecondStageProcessorTik op info to GE
    REGISTER_CUSTOM_OP("FasterrcnnSecondStageProcessorTik")
        .FrameworkType(TENSORFLOW)
        .OriginOpType("FasterRCNNSecondStagePostProcessorTik")
        .ParseParamsFn(AutoMappingFn)
        .ImplyType(ImplyType::TVM);
}  
