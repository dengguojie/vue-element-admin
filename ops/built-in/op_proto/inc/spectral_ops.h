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
 * @file spectral_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_OP_SPECTRAL_OPS_H
#define GE_OP_SPECTRAL_OPS_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Real-valued fast Fourier transform.

*@par Inputs:
*@li input: A float32 tensor.
*@li fft_length: An int32 tensor of shape [1]. The FFT length.

*@par Outputs:
*@li y: A complex64 tensor of the same rank as `input`. The inner-most \n
dimension of `input` is replaced with the `fft_length / 2 + 1` unique \n
frequency components of its 1D Fourier transform.

*@par Third-party framework compatibility
* Compatible with TensorFlow RFFT operator.
*/
REG_OP(RFFT)
    .INPUT(input, TensorType({DT_FLOAT}))
    .INPUT(fft_length, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_COMPLEX64}))
    .OP_END_FACTORY_REG(RFFT)

}  // namespace ge

#endif //GE_OP_SPECTRAL_OPS_H