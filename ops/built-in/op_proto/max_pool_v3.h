/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

/*!
 *\file max_pool_v3.h
 *\brief head file for max_pool_v3
 */
#ifndef GE_OP_MAX_POOLV3_H
#define GE_OP_MAX_POOLV3_H

#include "graph/operator_reg.h"

namespace ge {

REG_OP(MaxPoolV3)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_QINT8, DT_INT16, DT_UINT16, DT_INT32,
                          DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_QINT8, DT_INT16, DT_UINT16, DT_INT32,
                           DT_INT64}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(data_format, String, "NCHW")
    .ATTR(global_pooling, Bool, false)
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolV3)
}  // namespace ge

#endif  // GE_OP_MAX_POOLV3_H
