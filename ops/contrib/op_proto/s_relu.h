/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
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
 * @file s_relu.h
 *
 * @version 1.0
 */

#ifndef GE_OP_SRELU_H
#define GE_OP_SRELU_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief piecewise linear activation function
 *
 * @par Inputs:
 * 5 inputs, including:
 * @li x : input data. required
 * DataType: float16
 * Format: NC1HWC0
 *
 * @li tr : tr. required
 * DataType: float16
 * Format: NC1HWC0
 *
 * @li ar : ar. required
 * DataType: float16
 * Format: NC1HWC0
 *
 * @li tl : tl. required
 * DataType: float16
 * Format: NC1HWC0
 *
 * @li al : al. required
 * DataType: float16
 * Format: NC1HWC0
 *
 * @par Attributes:
 * @li channel_shared: whether the tr/ar/tl/al was shared in the channel. bool
 *
 * @par Outputs:
 * 1 outputs, including:
 * @li y : output data. required
 * DataType: float16
 * Format: NC1HWC0
 *
 * @attention Constraints:
 * @li channel_shared must be false
 * @li Dim(c) of x must be less than 30700
 */
REG_OP(SReLU)
    .INPUT(x, TensorType({DT_FLOAT16, }))
    .INPUT(tr, TensorType({DT_FLOAT16, }))
    .INPUT(ar, TensorType({DT_FLOAT16, }))
    .INPUT(tl, TensorType({DT_FLOAT16, }))
    .INPUT(al, TensorType({DT_FLOAT16, }))
    .OUTPUT(y, TensorType({DT_FLOAT16, }))
    .ATTR(channel_shared, Bool, false)
    .OP_END_FACTORY_REG(SReLU)
}
#endif // GE_OP_SRELU_H
