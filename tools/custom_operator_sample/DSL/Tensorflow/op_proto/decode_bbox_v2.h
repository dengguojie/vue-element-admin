/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file decode_bbox_v2.h
 * \brief
 */
#ifndef DECODE_BBOX_V2_H_
#define DECODE_BBOX_V2_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Computes decode bboxv2 function.
*
*@par Inputs:
*Inputs include:
* @li boxes: A Tensor. Must be float16 or float32.
* @li anchors: A Tensor. Must be int32.
*
*@par Attributes:
* @li scales: optional, listfloat, .
* @li decode_clip: optional, float, threahold of decode process.
* @li reversed_boxes: optional, bool,.
*
*@par Outputs:
* y: A Tensor. Must have the same type as box_predictions.
*/
REG_OP(DecodeBboxV2)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(anchors, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(scales, ListFloat, {1.0, 1.0, 1.0, 1.0})
    .ATTR(decode_clip, Float, 0.0)
    .ATTR(reversed_box, Bool, false)
    .OP_END_FACTORY_REG(DecodeBboxV2)

}  // namespace ge

#endif  // DECODE_BBOX_V2_H_