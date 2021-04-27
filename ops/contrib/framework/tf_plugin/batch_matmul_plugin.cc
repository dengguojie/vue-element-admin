/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: batch_matmul framework plugin cpp file
 * Author:
 * Create: 2020-6-11
 * Note:
 */

#include "register/register.h"

namespace domi {
    REGISTER_CUSTOM_OP("BatchMatmul")
        .FrameworkType(TENSORFLOW)
        .OriginOpType("BatchMatmulOP")
        .ParseParamsFn(AutoMappingFn)
        .ImplyType(ImplyType::TVM);
}
