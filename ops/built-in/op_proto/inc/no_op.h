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
 * @file  no_op.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_NO_OP_H_
#define GE_NO_OP_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Does nothing. Only useful as a placeholder for control edges.

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator NoOp.
*/

REG_OP(NoOp)
    .OP_END_FACTORY_REG(NoOp)

}  // namespace ge

#endif  // GE_NO_OP_H_
