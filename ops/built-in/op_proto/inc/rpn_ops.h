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
 * @file rpn_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 **/
#ifndef GE_OP_RPN_OPS_H
#define GE_OP_RPN_OPS_H

#include "graph/operator_reg.h"
namespace ge {
/**
*@brief Iteratively removes lower scoring boxes which have an IoU greater than
* iou_threshold with higher scoring box according to their
* intersection-over-union (IoU).

*@par Input:
* @li box_scores: 2-D tensor with shape of [N, 8], including proposal boxes and
* corresponding confidence scores.

* @par Attributes:
* @li iou_threshold: An optional float. The threshold for deciding whether boxes
* overlap too much with respect to IOU.

* @par Outputs:
* @li selected_boxes: 2-D tensor with shape of [N,5], representing filtered
* boxes including proposal boxes and corresponding confidence scores.
* @li selected_idx: 1-D tensor with shape of [N], representing the index of
* input proposal boxes.
* @li selected_mask: 1-D tensor with shape of [N], the symbol judging whether
* the output proposal boxes is valid.

* @attention Constraints:
* The 2nd-dim of input box_scores must be equal to 8.\n
* Only supports 2864 input boxes at one time.\n

*/
REG_OP(NMSWithMask)
    .INPUT(box_scores, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(selected_boxes, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(selected_idx, TensorType({DT_INT32}))
    .OUTPUT(selected_mask, TensorType({DT_UINT8}))
    .ATTR(iou_threshold, Float, 0.5)
    .OP_END_FACTORY_REG(NMSWithMask)
}  // namespace ge

#endif // GE_OP_TRAINING_OPS_H
