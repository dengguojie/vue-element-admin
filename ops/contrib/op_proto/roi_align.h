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
* @file roi_align.h
*
* @version 1.0
*/

#ifndef GE_OP_ROIALIGN_H
#define GE_OP_ROIALIGN_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief the implementation of roi align

* @par Inputs:
* 2 inputs, including:
* @li: feature map. Input RCNN feature map.
*    ParamType: required
*    DataType: float16
*    Format: NC1HWC0
*    Shape: Only support single Batch, N = 1
* @li: proposal, the proposal network's output.
*    ParamType: required
*    DataType: float16
*    Format: NC1HWC0
*    Shape: C must be 5, C1 = (Int)(C + 15) / 16 = 1.
*           H, W must be equal 1.

* @par Attributes:
* @li: pooled_h, output pooling height.
*    ParamType: required
*    DataType: float
* @li: pooled_w: output pooling width.
*    ParamType: required
*    DataType: float
* @li: spatial_scale
*    ParamType: optional
*    DataType: float
*    DefaultValue: 0.0625

* @par Outputs:
* 1 output, including:
* @li: roi_align
*    ParamType: required
*    DataType: float16
*    Format: NC1HWC0
*    Shape: N must be equal to proposal 's 1st dimsize.
*           H must be equal to pooled_h.
*           W must be equal to pooled_w.

* @attention Constraints:
 1. N of proposal * pooled_h * pooled_w < 304 * 8 * 8.
 2. (pooled_h * pooled_w + 15) // 16 * 16 < 4080.
 3. pooled_h(pooled_w) < H of input 2(W of input 2).

*/

REG_OP(ROIAlignTIK)
    .INPUT(feature_map, TensorType({DT_FLOAT16}))
    .INPUT(rois, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(pooled_h, Int)
    .REQUIRED_ATTR(pooled_w, Int)
    .ATTR(spatial_scale, Float, 0.0625)
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(ROIAlignTIK)
}

#endif // GE_OP_CONCAT_H
