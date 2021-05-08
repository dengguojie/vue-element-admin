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
 * @file tile.h
 *
 * @version 1.0
 */

#ifndef GE_OP_TILE_CAFFE_H
#define GE_OP_TILE_CAFFE_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Expands data in a specified dimension (N/C/H/W)
*
* @par Inputs:
*   1 inputs, including:
* @li x : input data. required
*        DataType: float16
*        Format: NC1HWC0
*
* @par Attributes:
* @li axis: the axis to expand. int
* @li tiles: the times to expand. int
*
* @par Outputs:
*   1 outputs, including:
* @li y : output data. required
*        DataType: float16
*        Format: NC1HWC0
*
* @attention Constraints:
* @li when axis == 1, the dim(c) of x must be less than 3840
*/
REG_OP(TileCaffe)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(axis, Int, 1)
    .ATTR(tiles, Int, 2)
    .OP_END_FACTORY_REG(TileCaffe)
}
#endif // GE_OP_TILE_CAFFE_H

