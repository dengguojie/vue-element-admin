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
 * @file clip.h
 *
 * @version 1.0
 */

#ifndef GE_OP_CLIP_H
#define GE_OP_CLIP_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Limits the value between min and max
 *
 * @par Inputs:
 *   1 inputs, including:
 * @li x : input data. required
 *       DataType: float16
 *       Format: NC1HWC0
 *
 * @par Attributes:
 * @li min: the min value. float
 * @li max: the max value. float
 *
 * @par Outputs:
 *   1 outputs, including:
 * @li y : output data. required
 *       DataType: float16
 *       Format: NC1HWC0
 *
 * @attention Constraints:
 * @li none
 */
REG_OP(Clip)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(min, Float, 0)
    .ATTR(max, Float, 0)
    .OP_END_FACTORY_REG(Clip)
}
#endif // GE_OP_CLIP_H

