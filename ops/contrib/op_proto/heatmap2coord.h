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
 * @file heatmap2coord.h
 *
 * @version 1.0
 */

#ifndef GE_OP_HEATMAP2COORD_H
#define GE_OP_HEATMAP2COORD_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief pick out the coordinate of the first max value in HW-plane
*
* @par Inputs:
*   1 inputs, including:
* @li x : input data. required
*        DataType: float16
*        Format: NC1HWC0
*
* @par Outputs:
*   1 outputs, including:
* @li y : output data. required
*        DataType: float16
*        Format: NC1HWC0
*
* @attention Constraints:
* @li input h,w <= 2048
* @li input data size should not bigger than 2GB
*/

REG_OP(Heatmap2Coord)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(Heatmap2Coord)
}
#endif // GE_OP_HEATMAP2COORD_H
