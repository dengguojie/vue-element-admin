/**
 * Copyright (C)  2020-2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_CONV2_D_H
#define GE_OP_CONV2_D_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(Conv2D)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT32}))
    .INPUT(filter, TensorType({DT_FLOAT16,DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT32}))
    .ATTR(strides, ListInt, {1, 1, 1, 1})
    .ATTR(pads, ListInt, {1, 1, 1, 1})
    .REQUIRED_ATTR(dilations, ListInt)
    .OP_END_FACTORY_REG(Conv2D)
}
#endif //GE_OP_CONV2_D_H
