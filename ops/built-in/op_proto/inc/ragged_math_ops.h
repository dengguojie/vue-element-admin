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
 * @file ragged_math_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_OP_RAGGED_MATH_OPS_H
#define GE_OP_RAGGED_MATH_OPS_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Returns a `RaggedTensor` containing the specified sequences of numbers.

*@par Inputs:
*@li starts: The starts of each range.
*@li limits: The limits of each range.
*@li deltas: The deltas of each range.

*@par Outputs:
*y:A Returns The `row_splits` for the returned `RaggedTensor`.The `flat_values` for the returned `RaggedTensor`.

*@attention Constraints: \n
*The input tensors `starts`, `limits`, and `deltas` may be scalars or vectors. \n
*The vector inputs must all have the same size.  Scalar inputs are broadcast \n
*to match the size of the vector inputs.

*@par Third-party framework compatibility
* Compatible with tensorflow RaggedRange operator.
*/

REG_OP(RaggedRange)
    .INPUT(starts, TensorType({DT_FLOAT,DT_DOUBLE,DT_INT32,DT_INT64}))
    .INPUT(limits, TensorType({DT_FLOAT,DT_DOUBLE,DT_INT32,DT_INT64}))
    .INPUT(deltas, TensorType({DT_FLOAT,DT_DOUBLE,DT_INT32,DT_INT64}))
    .OUTPUT(rt_nested_splits, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(rt_dense_values, TensorType({DT_FLOAT,DT_DOUBLE,DT_INT32,DT_INT64}))
    .REQUIRED_ATTR(Tsplits, Type)
    .OP_END_FACTORY_REG(RaggedRange)

}  // namespace ge

#endif //GE_OP_RAGGED_MATH_OPS_H