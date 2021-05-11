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
 * @file ceil.h
 *
 * @version 1.0
 */

#ifndef GE_OP_CEIL_H
#define GE_OP_CEIL_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief returns element-wise smallest integer not less than x.
 *
 * @par Inputs:
 * 1 inputs, including:
 * @li input_gm : input_data, required;
 * DataType: float16;
 * Format: NC1HWC0.
 *
 * @par Outputs:
 * 1 outputs, including:
 * @li output_gm : output_data, required;
 * DataType: float16;
 * Format: NC1HWC0.
 */
REG_OP(TikCeil)
    .INPUT(input_gm, TensorType({ DT_FLOAT16 }))
    .OUTPUT(output_gm, TensorType({ DT_FLOAT16 }))
    .OP_END_FACTORY_REG(TikCeil);
}

#endif // GE_OP_CEIL_H
