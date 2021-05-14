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
 * @file fasterrcnn_first_stage_processor_tik.h
 *
 * @version 1.0
 */

#ifndef GE_OP_FASTERRCNN_FIRST_STAGE_PROCESSOR_TIK_H
#define GE_OP_FASTERRCNN_FIRST_STAGE_PROCESSOR_TIK_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief withdraw foregrounds of feature map
 *
 * @par Inputs:
 * 3 inputs, including:
 * @li scorelist_gm : score for each anchor. required
 * DataType: float16
 * Format: NHWC
 * @li boxlist_gm : info to revise anchor. required
 * DataType: float16
 * Format: NHWC
 * @li anchor_gm : input grid anchor. required
 * DataType: float16
 * Format: NHWC
 *
 * @par Outputs:
 * 1 outputs, including:
 * @li output_gm : proposal of feature map. required
 * DataType: float16
 * Format: ND
 *
 * @attention Constraints:
 * @li only support one sets of input:
 * [[1, 29184, 2],[1, 29184, 1, 4],[1, 29184, 1, 4]]
 * [[1, 61440, 2],[1, 61440, 1, 4],[1, 61440, 1, 4]]
 */

// namespace ge
REG_OP(FasterrcnnFirstStageProcessorTik)
    .INPUT(scorelist_gm, TensorType({ DT_FLOAT16 }))
    .INPUT(boxlist_gm, TensorType({ DT_FLOAT16 }))
    .INPUT(anchorlist_gm, TensorType({ DT_FLOAT16 }))
    .OUTPUT(output_gm, TensorType({ DT_FLOAT16 }))
    .OP_END_FACTORY_REG(FasterrcnnFirstStageProcessorTik)
}

#endif // GE_OP_FASTERRCNN_FIRST_STAGE_PROCESSOR_TIK_H
