/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this
 * file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file p_sroi_align.h
 *
 * @version 1.0
 */

#ifndef GE_OP_PSROIALIGN_H
#define GE_OP_PSROIALIGN_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief the implementation of PSROIAlign

* @par Inputs:
* input 1: feature map, support NC1HWC0

* input 2: proposal,the proposal network's output, support NCHW

* @par Attributes:
* spatial_scale: use float
* outputdim: channel of output
* group_size: pooling width and height
* sample_num: the number of sample points

* @par Outputs:
* output 1: output_map, support NC1HWC0

* @attention Constraints:
* 1. outputdim = input featuremap 's channels/(groupsize*groupsize)
* 2. group_size < feature_map 's height and group_size < feautre_map 's width
* 3. PSROIAlign only support float16
* 4. sample_num ^ 2 % 8 == 0
**/
REG_OP(PSROIAlign)
    .INPUT(feature_map, TensorType({DT_FLOAT16}))
    .INPUT(rois, TensorType({DT_FLOAT16}))
    .OUTPUT(output_map, TensorType({DT_FLOAT16}))
    .ATTR(spatial_scale, Float, 0.0625)
    .REQUIRED_ATTR(output_dim, Int)
    .REQUIRED_ATTR(group_size, Int)
    .ATTR(sample_num, Int, 2)
    .OP_END_FACTORY_REG(PSROIAlign)
}

#endif // GE_OP_CONCAT_H
