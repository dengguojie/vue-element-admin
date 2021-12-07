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
 * \file batch_norm.h
 * \brief
 */
#ifndef BATCH_NORM_H
#define BATCH_NORM_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs batch normalization . \n

* @par Inputs:
* Five inputs, including: (NHWC, NCHW, or NC1HWC0 supported)
* @li x: A 4D or 5D Tensor of type float16 or float32, with format NHWC or NCHW for 4D or NC1HWC0 for 5D.
* @li scale: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW. Must be 5D
if input "x" is with format NC1HWC0. Specifies the scaling factor.
* @li offset: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW. Must be 5D
if input "x" is with format NC1HWC0. Specifies the offset.
* @li mean: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW. Must be 5D
if input "x" is with format NC1HWC0. Specifies the mean used for inference. Must be "None" if the
operation is used for training.
* @li variance: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW. Must be
5D if input "x" is with format NC1HWC0. Specifies the variance used for inference. Must be "None"
if the operation is used for training . \n

* @par Attributes:
* @li epsilon: An optional float32, specifying the small value added to variance to avoid dividing by zero. Defaults to "0.0001".
* @li data_format: An optional string, specifying the format of "x". Defaults to "NHWC".
* @li is_training: An optional bool, specifying if the operation is used for training or inference. Defaults to "True" . \n

* @par Outputs:
* Five outputs, including: (NHWC, NCHW, or NC1HWC0 supported)
* @li y: A 4D or 5D Tensor of type float16 or float32 for the normalized "x", with format NHWC or NCHW for 4D or NC1HWC0 for 5D.
* @li batch_mean: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW. Must be 5D
if input "x" is with format NC1HWC0. Specifies the mean of "x".
* @li batch_variance: A Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
Must be 5D if input "x" is with format NC1HWC0. Specifies the variance of "x".
* @li reserve_space_1: An optional Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
Must be 5D if input "x" is with format NC1HWC0. Specifies the mean of "x" for gradient computation. Pass "None" to skip this output.
* @li reserve_space_2: An optional Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
Must be 5D if input "x" is with format NC1HWC0. Specifies the variance of "x" for gradient computation. Pass "None" to skip this output . \n

* @attention Constraints:
* @li If the operation is used for inference and outputs "reserve_space_1" and "reserve_space_2" are available,
then "reserve_space_1" has the same value as "mean" and "reserve_space_2" has the same value as "variance".
* @li For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction . \n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator fused_batch_norm.
* @li Compatible with the TensorFlow operator fused_batch_norm_v2.
*/
REG_OP(BatchNorm)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_3, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNorm)
}  // namespace ge
#endif  // BATCH_NORM_H
