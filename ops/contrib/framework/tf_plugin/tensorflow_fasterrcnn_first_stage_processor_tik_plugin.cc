/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Plugin for Operator faster rcnn first stage scope processor
 * Author: 
 * Create: 2020-06-17
 */

#include "register/register.h"

namespace domi {
    // register FasterrcnnFirstStageProcessorTik op info to GE
    REGISTER_CUSTOM_OP("FasterrcnnFirstStageProcessorTik")
        .FrameworkType(TENSORFLOW)
        .OriginOpType("FasterRCNNFirstStageProcessorTik")
        .ParseParamsFn(AutoMappingFn)
        .ImplyType(ImplyType::TVM);
} 