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
 * @file resample.h
 *
 * @version 1.0
 */

#ifndef GE_OP_RESAMPLE_H
#define GE_OP_RESAMPLE_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief The N/C dimension remains unchanged, and zoom out or zoom in at the H/W dimension.
*
* @par Inputs:
*   1 inputs, including:
* @li x : input data. required
*        DataType: float16
*        Format: NC1HWC0
*
* @par Attributes:
* @li height: the height of y. int
* @li width: the width of y. int
* @li antialias: the antialias. bool
* @li type: the type of resample. int
*
* @par Outputs:
*   1 outputs, including:
* @li y : output data. required
*        DataType: float16
*        Format: NC1HWC0
*
* @attention Constraints:
* @li the H/W of the input and output must be less than 65535
* @li only support LINEAR and NEAREST
**/
REG_OP(Resample)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(height, Int, 1)
    .ATTR(width, Int, 1)
    .ATTR(antialias, Bool, true)
    .ATTR(type, Int, 1)
    .OP_END_FACTORY_REG(Resample)
}
#endif // GE_OP_RESAMPLE_H

