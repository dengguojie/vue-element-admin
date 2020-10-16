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
 * @file stateless_random_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_OP_STATELESS_RANDOM_OPS_H
#define GE_OP_STATELESS_RANDOM_OPS_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Draws samples from a multinomial distribution.

*@par Inputs:
include: \n
*@li logits:2-D Tensor with shape [batch_size, num_classes]. Each slice [i, :]\n
*represents the unnormalized log probabilities for all classes.
*@li num_samples:0-D. Number of independent samples to draw for each row slice.
*@li seed:The seed to generate random.

*@par Attributes:
*output_dtype:Output data type.

*@par Outputs:
*y:Output random number.

*@see StatelessMultinomial()

*@par Third-party framework compatibility
*compatible with StatelessMultinomial op of tensorflow
*/
REG_OP(StatelessMultinomial)
    .INPUT(logits, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .INPUT(num_samples, TensorType({DT_INT32}))
    .INPUT(seed, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(output_dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(StatelessMultinomial)

/**
*@brief Outputs deterministic pseudorandom random integers from a uniform distribution.

*@par Inputs:
*@li shape: The shape of the output tensor.
*@li seed: 2 seeds (shape [2]).
*@li minval: Minimum value (inclusive, scalar).
*@li maxval: Maximum value (exclusive, scalar).

*@par Outputs:
*y: Returns Random values with specified shape.

*@par Third-party framework compatibility
* Compatible with TensorFlow StatelessRandomUniformInt operator.
*/

REG_OP(StatelessRandomUniformInt)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(seed, TensorType({DT_INT32, DT_INT64}))
    .INPUT(minval, TensorType({DT_INT32, DT_INT64}))
    .INPUT(maxval, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(StatelessRandomUniformInt)

}  // namespace ge

#endif //GE_OP_STATELESS_RANDOM_OPS_H