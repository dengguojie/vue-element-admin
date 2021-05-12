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
 * @file fasterrcnn_map_tik.h
 *
 * @version 1.0
 */

#ifndef GE_OP_FASTERRCNN_MAP_TIK_H
#define GE_OP_FASTERRCNN_MAP_TIK_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief map absolute data to normalized data
 *
 * @par Inputs:
 * 1 inputs, including:
 * @li proposal_norm_gm : absolute data. required
 * DataType: float16
 * Format: ND
 *
 * @par Outputs:
 * 1 outputs, including:
 * @li proposal_abso_gm : normalized data. required
 * DataType: float16
 * Format: ND
 *
 * @attention Constraints:
 * @li two sets of input are valid
 * [[1, 100,4]];
 * [[1, 300,4]];
 */

    REG_OP(FasterrcnnMapTik)
        .INPUT(proposal_ori_gm, TensorType({ DT_FLOAT16 }))
        .OUTPUT(proposal_norm_gm, TensorType({ DT_FLOAT16 }))
        .OP_END_FACTORY_REG(FasterrcnnMapTik)
}

#endif // GE_OP_FASTERRCNN_MAP_TIK_H
