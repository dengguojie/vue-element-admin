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
 * @file ragged_array_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_OP_RAGGED_ARRAY_OPS_H
#define GE_OP_RAGGED_ARRAY_OPS_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Gather ragged slices from `params` axis `0` according to `indices`.

*@par Inputs:
*@li params_nested_splits: The `nested_row_splits` tensors that define the row-partitioning for the \n
*params` RaggedTensor input.
*@li params_dense_values: The `flat_values` for the `params` RaggedTensor. There was a terminology change \n
*at the python level from dense_values to flat_values, so dense_values is the \n
*deprecated name.
*@li indices: Indices in the outermost dimension of `params` of the values that should be \n
*gathered.
*@li OUTPUT_RAGGED_RANK: The ragged rank of the output RaggedTensor. `output_nested_splits` will contain \n
*this number of `row_splits` tensors. This value should equal \n
*`indices.shape.ndims + params.ragged_rank - 1`.

*@par Outputs:
*y:A Returns The `nested_row_splits` tensors that define the row-partitioning for the \n
*returned RaggedTensor.The `flat_values` for the returned RaggedTensor.

*@par Third-party framework compatibility
* Compatible with tensorflow RaggedGather operator.
*/

REG_OP(RaggedGather)
    .DYNAMIC_INPUT(params_nested_splits, TensorType({DT_INT32, DT_INT64}))
    .INPUT(params_dense_values, TensorType({DT_INT32, DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .DYNAMIC_OUTPUT(output_nested_splits, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(output_dense_values, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(Tsplits, Type)
    .ATTR(PARAMS_RAGGED_RANK, Int, 1)
    .ATTR(OUTPUT_RAGGED_RANK, Int, 0)
    .OP_END_FACTORY_REG(RaggedGather)

}  // namespace ge

#endif //GE_OP_RAGGED_ARRAY_OPS_H