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
 * @file swap_co_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_OP_SWAP_CO_OPS_H_
#define GE_OP_SWAP_CO_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Folds the convolution input weight constant of the preceding layer \n
* of PSROIPooling to convert the N dimension of the weight from \n
* (output_dim, group_size*group_size) to \n
* (group_size*group_size, int((output_dim+15)/C0)*C0).
*@see PSROIPooling

*@par Inputs:
* One input:
*x: An NCHW tensor of type float16 or float32, describing the weight of\n
* convolution. Dim N must equal output_dim*group_size*group_size.

*@par Attributes:
*@li output_dim: A required int32, specifying the number of output channels.\n
* Must be greater than "0".
*@li group_size: A required int32, specifying the number of groups to encode\n
* position-sensitive score maps. Must be within the range (0, 128).

*@par Outputs:
*y: An NCHW tensor of type float16 or float32, describing the result weight\n
* of convolution.
*/

REG_OP(SwapCo)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(output_dim, Int, 0)
    .ATTR(group_size, Int, 0)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(SwapCo)

}  // namespace ge

#endif  // GE_OP_SWAP_CO_OPS_H_
