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
 * @file parsing_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_OP_PARSING_OPS_H
#define GE_OP_PARSING_OPS_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Converts each string in the input Tensor to the specified numeric type.

*@par Inputs:
*Inputs include: \n
*x: A Tensor. Must be one of the following types: string.

*@par Attributes:
*out_type: The numeric type to interpret each string in string_tensor as.

*@par Outputs:
*y: A Tensor. Has the same type as x.

*@attention Constraints:\n
*-The implementation for StringToNumber on Ascend uses AICPU, with bad performance.\n

*@par Third-party framework compatibility
*@li compatible with tensorflow StringToNumber operator.
*/
REG_OP(StringToNumber)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .ATTR(out_type, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(StringToNumber)

}  // namespace ge

#endif  // GE_OP_PARSING_OPS_H
