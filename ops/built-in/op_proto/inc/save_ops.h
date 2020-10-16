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
 * @file save_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_SAVE_OPS_H_
#define GE_OP_SAVE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(Save)
    .DYNAMIC_INPUT(tensors, TensorType:ALL())
    .OP_END_FACTORY_REG(Save)

} // namespace ge


#endif  // GE_OP_SAVE_OPS_H_