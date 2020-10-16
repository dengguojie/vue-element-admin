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
 * @file reduce_ZEROSLIKE.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_ZEROSLIKE_H
#define GE_OP_ZEROSLIKE_H

#include "../graph/operator_reg.h"
namespace ge {

REG_OP(Zeroslike)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .ATTR(dst_dtype, String, "")
    .ATTR(optimize, Bool, true) // only surport true
    .INFER_SHAPE_AND_TYPE(ELMTWISE_INFER("x", "y"))
    .OP_END_FACTORY_REG(Zeroslike)
} // namespace ge

#endif // GE_OP_REDUCE_ZEROSLIKE_H