/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file tik_rank.h
 *
 * @version 1.0
 */

#ifndef GE_OP_RANK_H
#define GE_OP_RANK_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief return the order of input
 *
 * @par Inputs:
 *  1 inputs, including:
 * @li input_gm, input data, required;
 *      DataType: float,float16,int32;
 *      Format: ND,ND,ND.
 *
 * @par Outputs:
 *  1 output, including:
 * @li output_gm: output data, required;
 *      DataType: int32,int32,int32;
 *      Format: ND,ND,ND.
 *
 */
    REG_OP(TikRank)
        .INPUT(input_gm, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT32 }))
        .OUTPUT(output_gm, TensorType({ DT_INT32 }))
        .OP_END_FACTORY_REG(TikRank);
}

#endif // GE_OP_RANK_H
