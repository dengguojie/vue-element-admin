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
 * @file  condtake_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_OP_CONDTAKE_OPS_H_
#define GE_OP_CONDTAKE_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Take elements from data if specific condition is satisfied on mask.

*@par Inputs:
*@li data: input tensor from which to take elements, High-dimension input would \n
first be flattened.
*@li mask: condition param; must be the same shape with data.

*@par Attributes:
*@li mode:convert by convert in Mode.
*@li val:convert by <class 'float'>
*@li eps:convert by <class 'float'> (default: 1e-06)

*@par Outputs:
*@li out_data: the elements taken
*@li out_index: the indices corresponding to those elements
*@li valid_num: elements of out_data and out_index from zeros to valid_num is valid.
*/

REG_OP(CondTake)
    .INPUT(data, TensorType({DT_FLOAT}))
    .INPUT(mask, TensorType({DT_FLOAT}))
    .OUTPUT(out_data, TensorType({DT_FLOAT}))
    .OUTPUT(out_index, TensorType({DT_INT32}))
    .OUTPUT(valid_num, TensorType({DT_INT32}))
    .REQUIRED_ATTR(mode, String)
    .REQUIRED_ATTR(val, Float)
    .ATTR(eps, Float, 1e-06)
    .OP_END_FACTORY_REG(CondTake)
}  // namespace ge

#endif  // GE_OP_ARRAY_OPS_H_
