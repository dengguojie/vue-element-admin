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
 * @file resource_variable_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_OP_RESOURCE_VARIABLE_OPS_H
#define GE_OP_RESOURCE_VARIABLE_OPS_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

REG_OP(VarHandleOp)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(shape, ListInt, ge::UNKNOWN_SHAPE)
    .OUTPUT(y, TensorType({DT_RESOURCE}))
    .OP_END_FACTORY_REG(VarHandleOp)

REG_OP(AssignVariableOp)
    .INPUT(resource, TensorType({DT_RESOURCE}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AssignVariableOp)

REG_OP(AssignAddVariableOp)
    .INPUT(resource, TensorType({DT_RESOURCE}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AssignAddVariableOp)

REG_OP(AssignSubVariableOp)
    .INPUT(resource, TensorType({DT_RESOURCE}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AssignSubVariableOp)

}  // namespace ge

#endif //GE_OP_RESOURCE_VARIABLE_OPS_H