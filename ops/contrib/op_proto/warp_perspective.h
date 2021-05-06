/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file warp_perspective.h
 *
 * @version 1.0
 */

#ifndef GE_OP_WARPPERSPECTIVE_H
#define GE_OP_WARPPERSPECTIVE_H

#include "graph/operator_reg.h"
/**
 * @brief image perspective transformation
 *
 * @par Inputs:
 * 2 inputs, including:
 * @li x: Src Image to be transform, a NC1HWC0 tensor of type is: float16
 * @li matrix: transformation matrix, a NC1HWC0 tensor of type is: \n
 * float16
 *
 * @par Attributes:
 * @li dst_height: constant value, required, dtype=int, \n
 * dst_height is less than or equals to 2048.
 * @li constant_value: constant value, required, dtype=int, \n
 * dst_width is less than or equals to 2048.
 * @li interpolation: interpolation method, required, dtype=string, \n
 * value = "INTER_LINEAR" or "INTER_NEAREST"
 * @li constant_value: constant value, required, dtype=int, \n
 * constant_value is less than or equals to 2048.
 *
 * @par Outputs:
 * 1 outputs, including:
 * @li y: Dst Image after transform , a NC1HWC0 tensor of type is float16
 *
 * @attention Constraints:
 * @li input h, w < 2048
 * @li limit to one batch compute
 * @li dim of C channel shall be either 1 or 3
 */
namespace ge {
REG_OP(WarpPerspective)
    .INPUT(x, TensorType({ DT_FLOAT16 }))
    .INPUT(matrix, TensorType({ DT_FLOAT16 }))
    .OUTPUT(y, TensorType({ DT_FLOAT16 }))
    .REQUIRED_ATTR(dst_height, Int)
    .REQUIRED_ATTR(dst_width, Int)
    .REQUIRED_ATTR(interpolation, String)
    .REQUIRED_ATTR(constant_value, Int)
    .OP_END_FACTORY_REG(WarpPerspective)
}

#endif
