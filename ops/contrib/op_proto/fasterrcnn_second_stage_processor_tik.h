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
 * @file fasterrcnn_second_stage_processor_tik.h
 *
 * @version 1.0
 */

#ifndef GE_OP_FASTERRCNN_Second_STAGE_PROCESSOR_TIK_H
#define GE_OP_FASTERRCNN_Second_STAGE_PROCESSOR_TIK_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief second stage of processing fasterrcnn algorithm,
 * compute the score of each proposal of each class
 * @par Inputs:
 * 3 inputs, including:
 * @li proposal_boxes_gm : coordinates of proposal. required
 * DataType: float16
 * Format: ND
 * @li box_encodings_gm : feature map. required
 * DataType: float16
 * Format: ND
 * @li scorelist_gm : input scorelist of each class. required
 * DataType: float16
 * Format: ND
 *
 * @par Outputs:
 * 5 outputs, including:
 * @li score_out_gm : predicted score of each proposal. required
 * DataType: float16
 * Format: ND
 * @li output_class : detected and output class of each proposal. required
 * DataType: float16
 * Format: ND
 * @li output_boxes : detected and parsed coordinates of each proposal. required
 * DataType: float16
 * Format: ND
 * @li output_num_detection : detected and parsed output number. required
 * DataType: float16
 * Format: ND
 * @li output_score : detect and parse the score of class of each proposal. required
 * DataType: float16
 * Format: ND
 *
 * @attention Constraints:
 * @li only two sets of input are valid:
 * [[1, 100,4],[100, 360],[100 ,91]]
 * [[1, 300,4],[300, 8],[300 ,3]]
 */

    REG_OP(FasterrcnnSecondStageProcessorTik)
        .INPUT(box_encodings_gm, TensorType({ DT_FLOAT16 }))
        .INPUT(proposal_boxes_gm, TensorType({ DT_FLOAT16 }))
        .INPUT(scorelist_gm, TensorType({ DT_FLOAT16 }))
        .OUTPUT(score_out_gm, TensorType({ DT_FLOAT16 }))
        .OUTPUT(output_class, TensorType({ DT_FLOAT16 }))
        .OUTPUT(output_boxes, TensorType({ DT_FLOAT16 }))
        .OUTPUT(output_num_detection, TensorType({ DT_FLOAT16 }))
        .OUTPUT(output_score, TensorType({ DT_FLOAT16 }))
        .OP_END_FACTORY_REG(FasterrcnnSecondStageProcessorTik)
}

#endif // GE_OP_FASTERRCNN_Second_STAGE_PROCESSOR_TIK_H
