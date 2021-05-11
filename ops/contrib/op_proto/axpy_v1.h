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
 * @file axpy_v1.h
 *
 * @version 1.0
 */

#ifndef GE_OP_AXPY_V1_H
#define GE_OP_AXPY_V1_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief calculate F = a * X + Y
 *
 * @par Inputs:
 * Three inputs, including:
 * @li input_a : Format is NC1HWC0. Must be one of the following types: \n
 * float16. Shape of it can be (N,C) or (N,C,1,1).
 * @li input_x : Format is NC1HWC0. Types of it must be same as input_a.
 * @li input_y : Format is NC1HWC0. Types of it must be same as input_a.
 *
 * @par Outputs:
 * @li output_z : Format is NC1HWC0. Types of it must be same as input_a.
 *
 * @attention Constraints:
 * @li dim N and C of input_a and input_x must be consistent
 * @li shape of input_x and input_y, output_z must be consistent
 */
REG_OP(AxpyV1)
    .INPUT(input_a, TensorType({ DT_FLOAT16, DT_FLOAT, DT_INT32 }))
    .INPUT(input_x, TensorType({ DT_FLOAT16, DT_FLOAT, DT_INT32 }))
    .INPUT(input_y, TensorType({ DT_FLOAT16, DT_FLOAT, DT_INT32 }))
    .OUTPUT(output_z, TensorType({ DT_FLOAT16, DT_FLOAT, DT_INT32 }))
    .OP_END_FACTORY_REG(AxpyV1)
}

#endif // GE_OP_AXPY_V1_H
