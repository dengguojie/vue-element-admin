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
 * @file interpv2.h
 *
 * @version 1.0
 */

#ifndef GE_OP_INTERPV2_H
#define GE_OP_INTERPV2_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief image interpolation using resize bilinear
 *
 * @par Inputs:
 * 2 inputs, including:
 * @li input_0 : Image to be interpolated, an NC1HWC0 tensor, required;
 *      DataType: float16.
 * @li input_1 : Define output shape, an NC1HWC0 tensor, required;
 *      DataType: float16.
 *
 * @par Outputs:
 * 1 outputs, including:
 * @li output : Image after interpolate, an NC1HWC0 tensor, required;
 *      DataType:  float16.
 *
 * @attention Constraints:
 * @li dim in n and c of two input should be same.
 * @li the product of dim of N and dim of C shall be less than 128 if is \n
 *      not aligned with 128.
 * @li dim in h and w of two input dim should not bigger than 64.
 */
REG_OP(Interpv2)
    .INPUT(input_0, TensorType({ DT_FLOAT16 }))
    .INPUT(input_1, TensorType({ DT_FLOAT16 }))
    .OUTPUT(output, TensorType({ DT_FLOAT16 }))
    .OP_END_FACTORY_REG(Interpv2)
}
#endif // GE_OP_INTERPV2_H
