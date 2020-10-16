/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file bitwise_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_OP_BITWISE_OPS_H_
#define GE_OP_BITWISE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Element-wise computes the bitwise right-shift of x and y.

*@par Inputs: 
*Input "x" is a k-dimensional tensor. Inputs "num_lower" and "num_upper" \n
are 0D scalars.
* @li x: A Tensor. Must be one of the following types: int8, int16, int32, \n
int64, uint8, uint16, uint32, uint64. \n
* @li y: A Tensor. Has the same type as "x". \n

*@par Outputs: 
* z: A Tensor. Has the same type as "x". \n

*@attention Constraints: \n
*Unique runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator RightShift.
*/

REG_OP(RightShift)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .INPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(z, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
            DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OP_END_FACTORY_REG(RightShift)

}  // namespace ge

#endif  // GE_OP_BITWISE_OPS_H_
