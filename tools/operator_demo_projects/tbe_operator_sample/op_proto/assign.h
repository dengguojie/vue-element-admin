/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Description: Huawei Code
 *
 * Author: Huawei
 *
 * Create: 2020-01-01
 *
 */
#ifndef GE_OP_ASSIGN_H
#define GE_OP_ASSIGN_H
#include "graph/operator_reg.h"

namespace ge {
REG_OP(Assign)
    .INPUT(ref, TensorType::BasicType())
    .INPUT(value, TensorType::BasicType())
    .OUTPUT(ref, TensorType::BasicType())
    .OP_END_FACTORY_REG(Assign)
} // namespace ge

#endif // GE_OP_ASSIGN_H
