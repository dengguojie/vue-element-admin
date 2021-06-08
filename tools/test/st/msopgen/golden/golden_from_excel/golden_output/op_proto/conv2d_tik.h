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

#ifndef GE_OP_CONV2D_TIK_H
#define GE_OP_CONV2D_TIK_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(Conv2DTik)
    .INPUT(x1, TensorType({DT_FLOAT,DT_FLOAT16}))
    .INPUT(x2, TensorType({DT_FLOAT,DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT,DT_FLOAT16}))
    .ATTR(strides, ListInt, [1,1,1,1])
    .ATTR(pads, ListInt, [1,1,1,1])
    .ATTR(dilations, ListInt, [1,1,1,1])
    .OP_END_FACTORY_REG(Conv2DTik)
}
#endif //GE_OP_CONV2D_TIK_H
