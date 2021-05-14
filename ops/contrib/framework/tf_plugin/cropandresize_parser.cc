/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Plugin for Operator CropAndResize
 * Author: 
 * Create: 2020-06-17
 */

#include "register/register.h"

using namespace ge;

// namespace domi
namespace domi {
    REGISTER_CUSTOM_OP("FasterrcnnCropandresizeTik")
        .FrameworkType(TENSORFLOW)
        .OriginOpType("FasterRCNNCropAndResizeTik")
        .ParseParamsFn(AutoMappingFn)
        .ImplyType(ImplyType::TVM);
}
