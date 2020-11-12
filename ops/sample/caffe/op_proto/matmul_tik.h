/**
 * Copyright (C)  2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file matmul_tik.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_OP_MATMULTIK_H
#define GE_OP_MATMULTIK_H

#include "graph/operator_reg.h"

namespace ge {
REG_OP(MatmulTik)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))            
    .OP_END_FACTORY_REG(MatmulTik)
}

#endif  //GE_OP_MATMULTIK_H