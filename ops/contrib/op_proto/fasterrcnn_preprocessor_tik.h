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
 * @file fasterrcnn_preprocessor_tik.h
 *
 * @version 1.0
 */

#ifndef GE_OP_FASTERRCNN_PREPROCESSOR_TIK_H
#define GE_OP_FASTERRCNN_PREPROCESSOR_TIK_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief preprocess input graph
 *
 * @par Inputs:
 * 1 inputs, including:
 * @li data_a_gm : input graph. required
 * DataType: float16
 * Format: NHWC
 *
 * @par Outputs:
 * 1 outputs, including:
 * @li dst_gm : processed graph. required
 * DataType: float16
 * Format: NHWC
 *
 * @attention Constraints:
 * @li only support input graph of {{1, 300, 960, 3}}
 */

REG_OP(FasterrcnnPreprocessorTik)
    .INPUT(data_a_gm, TensorType({DT_FLOAT16}))
    .OUTPUT(dst_gm, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(FasterrcnnPreprocessorTik)
}

#endif // GE_OP_FASTERRCNN_PREPROCESSOR_TIK_H
