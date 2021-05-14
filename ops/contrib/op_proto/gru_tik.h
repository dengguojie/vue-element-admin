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
 * @file gru_tik.h
 *
 * @version 1.0
 */

#ifndef GE_OP_CTC_GREEDY_DECODER_H
#define GE_OP_CTC_GREEDY_DECODER_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Compute Gated Recurrent Unit which is a fuse Operator
 *
 * @par Inputs:
 * 2 inputs, including:
 * @li input : 3-D Tensor sized (batch, timestep, feature), required;
 * DataType: float16;
 * Format: ND
 * @li input_weight : input weight, required;
 * DataType: float16;
 * Format: ND
 *
 * @par Outputs:
 * 1 outputs, including:
 * @li output : 3-D Tensor, required;
 * DataType: float16;
 * Format: ND \n
 * if fusegru=false, output size is (batch, timestep, units) \n
 * if fusegru=true, output size is (batch, timestep, units * 2) \n
 *
 *
 * @par Attributes:
 * @li units: Positive integer, dimensionality of the output space, required \n
 * only support 128
 * @li go_backwards: Boolean (default false). If true, process the backward gru
 * @li fusegru: Boolean (default true) \n
 * If true, process forword gru and backword gru, and concat them \n
 * If false, process single gru \n
 * @attention Constraints:
 * @li units only supoort 128
 * @li input only suport two shapes:(1, 46, 256), (1, 59, 256)
 * @li The value of input_weight should bewteen (-0.3, 0.3)
 */

REG_OP(GruTik)
    .INPUT(input, TensorType({ DT_FLOAT16 }))
    .INPUT(input_weight, TensorType({ DT_FLOAT16 }))
    .OUTPUT(output, TensorType({ DT_FLOAT16 }))
    .REQUIRED_ATTR(units, Int)
    .ATTR(go_backwards, Bool, false)
    .ATTR(fusegru, Bool, true)
    .OP_END_FACTORY_REG(GruTik)
}

#endif // GE_OP_CTC_GREEDY_DECODER_H
