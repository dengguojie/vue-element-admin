/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file stateless_random_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_STATELESS_RANDOM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STATELESS_RANDOM_OPS_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Draws samples from a multinomial distribution . \n

*@par Inputs:
include:
*@li logits:2-D Tensor with shape [batch_size, num_classes]. Each slice [i, :]
*represents the unnormalized log probabilities for all classes.
*@li num_samples:0-D. Number of independent samples to draw for each row slice.
*@li seed:The seed to generate random . \n

*@par Attributes:
*output_dtype:Output data type . \n

*@par Outputs:
*y:Output random number . \n

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
*@brief Outputs deterministic pseudorandom random integers from a uniform distribution . \n

*@par Inputs:
*@li shape: The shape of the output tensor.
*@li seed: 2 seeds (shape [2]).
*@li minval: Minimum value (inclusive, scalar).
*@li maxval: Maximum value (exclusive, scalar) . \n

*@par Outputs:
*y: Returns Random values with specified shape . \n

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

/**
* @brief Outputs random values from a normal distribution. \n

* @par Inputs:
* Inputs include:
* @li shape: A Tensor. Must be one of the following types: int32, int64.
      The shape of the output tensor. Batches are indexed by the 0th dimension.
* @li seed: 2 seeds (shape [2]).
* @li means: A Tensor. Must be one of the following types: half, bfloat16, float32, float64.
* @li stdevs: A Tensor. Must have the same type as means.
* @li min: A Tensor. Must have the same type as means. The minimum cutoff. May be -infinity.
* @li max: A Tensor. Must have the same type as means. \n

* @par Outputs:
* y: A Tensor. Has the same type as means. \n

* @attention Constraints:
* The implementation for StatelessParameterizedTruncatedNormal on Ascend uses AICPU, with bad performance. \n

* @par Third-party framework compatibility
* @li compatible with tensorflow StatelessParameterizedTruncatedNormal operator.
*/

REG_OP(StatelessParameterizedTruncatedNormal)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(seed, TensorType({DT_INT32, DT_INT64}))
    .INPUT(means, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(stdevs, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(min, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(max, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(StatelessParameterizedTruncatedNormal)

/**
* @brief Generate a single randomly distorted bounding box for an image . \n

* @par Inputs:
* Input images must be a 4-D tensor. Inputs include:
* @li image_size: 1-D, containing [height, width, channels].
* @li bounding_boxes: 3-D with shape [batch, N, 4] describing the N bounding
 boxes associated with the image.
* @li min_object_covered: The cropped area of the image must contain at least
 this fraction of any bounding box supplied. The value of this parameter should
 be non-negative. In the case of 0, the cropped area does not need to overlap
 any of the bounding boxes supplied .
* @li seed: A shape [2] Tensor, the seed to the random number generator. \n

* @par Attributes:
* @li aspect_ratio_range: The cropped area of the image must have an aspect
 ratio = width / height within this range.
* @li area_range: An optional list of `floats`. Defaults to `[0.05, 1]`. The
 cropped area of the image must contain a fraction of the supplied image
 within this range.
* @li max_attempts: Number of attempts at generating a cropped region of the
 image of the specified constraints. After max_attempts failures, return the
 entire image.
* @li use_image_if_no_bounding_boxes: Controls behavior if no bounding boxes
 supplied. If true, assume an implicit bounding box covering the whole input.
 If false, raise an error . \n

* @par Outputs:
* @li begin: 1-D, containing [offset_height, offset_width, 0].
* @li size: 1-D, containing [target_height, target_width, -1].
* @li bboxes: 3-D with shape [1, 1, 4] containing the distorted bounding box . \n

* @attention Constraints:
* Input images can be of different types but output images are always float . \n

* @par Third-party framework compatibility
* Compatible with tensorflow StatelessSampleDistortedBoundingBox operator.
*/

REG_OP(StatelessSampleDistortedBoundingBox)
    .INPUT(image_size, TensorType({ DT_UINT8, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64 }))
    .INPUT(bounding_boxes, TensorType({ DT_FLOAT }))
    .INPUT(min_object_covered, TensorType({ DT_FLOAT }))
    .INPUT(seed, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(begin, TensorType({ DT_UINT8, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64 }))
    .OUTPUT(size, TensorType({ DT_UINT8, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64 }))
    .OUTPUT(bboxes, TensorType({ DT_FLOAT }))
    .ATTR(aspect_ratio_range, ListFloat, { 0.75f, 1.33f })
    .ATTR(area_range, ListFloat, { 0.05f, 1.0f })
    .ATTR(max_attempts, Int, 100)
    .ATTR(use_image_if_no_bounding_boxes, Bool, false)
    .OP_END_FACTORY_REG(StatelessSampleDistortedBoundingBox)

/**
* @brief Outputs random values from a truncated normal distribution. \n

* @par Inputs:
* Inputs include:
* @li shape: A Tensor. Must be one of the following types: int32, int64. \n
* @li key: Key of RNG algorithm. Shape[1]. \n
* @li counter: Counter of RNG algorithm. Shape[2] for philox, shape[1] for threefry. \n
* @li alg: RNG algorithm. 1：philox 2：threefry. \n

* @par Attributes:
* @li dtype: dtype: A optional attr, specifying the output data type. Defaults to "DT_FLOAT". \n

* @par Outputs:
* y: A Tensor of types: float16, float32, double. A tensor of the specified shape
 filled with random truncated normal values. \n

* @attention Constraints:
* The implementation for StatelessTruncatedNormalV2 on Ascend uses AICPU, with bad performance.

* @par Third-party framework compatibility
* @li compatible with tensorflow StatelessTruncatedNormalV2 operator.
*/

REG_OP(StatelessTruncatedNormalV2)
    .INPUT(shape, TensorType({ DT_INT32, DT_INT64 }))
    .INPUT(key, TensorType({ DT_UINT64 }))
    .INPUT(counter, TensorType({ DT_UINT64 }))
    .INPUT(alg, TensorType({ DT_INT32 }))
    .OUTPUT(y, TensorType({ DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(StatelessTruncatedNormalV2)

/**
* @brief Outputs deterministic pseudorandom random numbers from a gamma distribution. \n

* @par Inputs:
* @li shape: The shape of the output tensor.
* @li seed: 2 seeds (shape [2]).
* @li alpha: The concentration of the gamma distribution. Shape must match the rightmost dimensions of shape. \n

* @par Outputs:
* y: A Tensor. Has the same type as alpha. \n

* @par Third-party framework compatibility
* Compatible with TensorFlow StatelessRandomGammaV2 operator.
*/

REG_OP(StatelessRandomGammaV2)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(seed, TensorType({DT_INT32, DT_INT64}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE}))
    .OP_END_FACTORY_REG(StatelessRandomGammaV2)

/**
* @brief Outputs deterministic pseudorandom random integers from a uniform distribution . \n

* @par Inputs:
* @li shape: The shape of the output tensor.
* @li seed: 2 seeds (shape [2]). \n

* @par Attributes:
* dtype:Output data type . \n

* @par Outputs:
* y: Returns Random values with specified shape . \n

* @par Third-party framework compatibility
* Compatible with TensorFlow StatelessRandomUniformFullInt operator.
*/

REG_OP(StatelessRandomUniformFullInt)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(seed, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}))
    .ATTR(dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(StatelessRandomUniformFullInt)

/**
* @brief Outputs deterministic pseudorandom random integers from a uniform distribution . \n

* @par Inputs:
* @li shape: The shape of the output tensor.
* @li key: Key for the counter-based RNG algorithm.
* @li counter: Initial counter for the counter-based RNG algorithm.
* @li alg: 0-D. The RNG algorithm. \n

* @par Attributes:
* dtype:Output data type . \n

* @par Outputs:
* y: Returns Random values with specified shape . \n

* @par Third-party framework compatibility
* Compatible with TensorFlow StatelessRandomUniformFullIntV2 operator.
*/

REG_OP(StatelessRandomUniformFullIntV2)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(key, TensorType({DT_UINT64}))
    .INPUT(counter, TensorType({DT_UINT64}))
    .INPUT(alg, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}))
    .ATTR(dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(StatelessRandomUniformFullIntV2)

/**
* @brief Outputs deterministic pseudorandom random integers from a uniform distribution . \n

* @par Inputs:
* @li shape: The shape of the output tensor.
* @li key: Key for the counter-based RNG algorithm.
* @li counter: Initial counter for the counter-based RNG algorithm.
* @li alg: 0-D. The RNG algorithm.
* @li minval: Minimum value (inclusive, scalar).
* @li maxval: Maximum value (exclusive, scalar) . \n

* @par Outputs:
* y: Returns Random values with specified shape . \n

* @par Third-party framework compatibility
* Compatible with TensorFlow StatelessRandomUniformIntV2 operator.
*/

REG_OP(StatelessRandomUniformIntV2)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(key, TensorType({DT_UINT64}))
    .INPUT(counter, TensorType({DT_UINT64}))
    .INPUT(alg, TensorType({DT_INT32}))
    .INPUT(minval, TensorType({DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}))
    .INPUT(maxval, TensorType({DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}))
    .OP_END_FACTORY_REG(StatelessRandomUniformIntV2)

/**
* @brief Outputs deterministic pseudorandom random integers from a binomial distribution. \n

* @par Inputs:
* @li shape: The shape of the output tensor.
* @li seed: 2 seeds (shape [2]).
* @li counts: The counts of the binomial distribution. Must be broadcastable with probs,
* and broadcastable with the rightmost dimensions of shape.
* @li probs: The probability of success for the binomial distribution.
* Must be broadcastable with counts and broadcastable with the rightmost dimensions of shape. \n

* @par Attributes:
* @li dtype: A optional int32, specifying the output data type. Defaults to "DT_INT32". \n

* @par Outputs:
* @li y: Returns Random values with specified shape. \n

* @par Third-party framework compatibility
* Compatible with TensorFlow StatelessRandomBinomial operator.
*/
REG_OP(StatelessRandomBinomial)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(seed, TensorType({DT_INT32, DT_INT64}))
    .INPUT(counts, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .INPUT(probs, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(StatelessRandomBinomial)

/**
* @brief Outputs deterministic pseudorandom random integers from a poisson distribution . \n

* @par Inputs:
* @li shape: The shape of the output tensor.
* @li seed: 2 seeds (shape [2]).
* @li lam: mean value value of poisson distribution . \n

* @par Attributes:
* dtype:Output data type . \n

* @par Outputs:
* y: Returns Random values with specified shape . \n

* @par Third-party framework compatibility
* Compatible with TensorFlow StatelessRandomUniformInt operator.
*/

REG_OP(StatelessRandomPoisson)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(seed, TensorType({DT_INT32, DT_INT64}))
    .INPUT(lam, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(StatelessRandomPoisson)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_STATELESS_RANDOM_OPS_H_
