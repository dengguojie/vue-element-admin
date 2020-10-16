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
 * @file npu_loss_scale_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_NN_LOSS_SCALE_OPS_H
#define GE_OP_NN_LOSS_SCALE_OPS_H
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Computes NPU alloc float status operator function.

*@par Outputs:
*data: A Tensor of data value. Must be float32.
*/
REG_OP(NPUAllocFloatStatusOperator)
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUAllocFloatStatusOperator)

/**
*@brief Computes NPU clear float status operator function.

*@par Inputs:
*addr: A Tensor of data memory address. Must be float32.

*@par Outputs:
*data: A Tensor of data value. Must be float32.
*/
REG_OP(NPUClearFloatStatusOperator)
    .INPUT(addr, TensorType{DT_FLOAT})
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUClearFloatStatusOperator)

/**
*@brief Computes NPU get float status operator function.

*@par Inputs:
*addr: A Tensor of data memory address. Must be float32.

*@par Outputs:
*data: A Tensor of data value. Must be float32.
*/
REG_OP(NPUGetFloatStatusOperator)
    .INPUT(addr, TensorType{DT_FLOAT})
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUGetFloatStatusOperator)

/**
*@brief Produces a variable with 0 in memory.

*@par Outputs:
*y: A Tensor of type int32, output eight numbers with a value of zero.
*/
REG_OP(NPUAllocFloatStatus)
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUAllocFloatStatus)

/**
*@brief Set the value of address 0x40000 to 0 in each core.

*@par Inputs:
*addr: A tensor of type float32.

*@par Outputs:
*data: A Tensor of type float32.
*/
REG_OP(NPUClearFloatStatus)
    .INPUT(addr, TensorType{DT_FLOAT})
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUClearFloatStatus)

/**
*@brief Get the value of address 0x40000.

*@par Inputs:
*addr: A tensor of type float32.

*@par Outputs:
*data: A Tensor of type float32.
*/
REG_OP(NPUGetFloatStatus)
    .INPUT(addr, TensorType{DT_FLOAT})
    .OUTPUT(data, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NPUGetFloatStatus)
}  // namespace ge

#endif  // GE_OP_NN_LOSS_SCALE_OPS_H
