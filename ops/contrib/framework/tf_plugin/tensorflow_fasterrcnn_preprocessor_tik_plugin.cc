/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Plugin for Operator faster rcnn preprocessor
 * Author: 
 * Create: 2020-06-17
 */

#include "register/register.h"

namespace domi {
    // register FasterrcnnMap1Tik op info to GE
    REGISTER_CUSTOM_OP("FasterrcnnPreprocessorTik")
        .FrameworkType(TENSORFLOW)
        .OriginOpType("FasterRCNNPreProcessorTik")
        .ParseParamsFn(AutoMappingFn)
        .ImplyType(ImplyType::TVM);
}  
