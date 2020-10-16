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
 * @file  warp_perspective_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_OP_WARP_PERSPECTIVE_OPS_H_
#define GE_OP_WARP_PERSPECTIVE_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Applies a perspective transformation to an image.

*@par Inputs:
*@li x: input tensor, format NCHW, type must be float.
*@li matrix: transformation matrix, format ND , shape must be (N, 9), type must be float.

*@par Attributes:
*@li out_height:output height.
*@li out_width:output width.
*@li borderType:border processing way, only support BORDER_CONSTANT and BORDER_REPLICATE, default BORDER_CONSTANT.
*@li constant: border processed value when borderType is BORDER_CONSTANT.

*@par Outputs:
*@li y: output tensor, format NCHW, type must be float.
*/

REG_OP(WarpPerspective)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(matrix, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(out_height, Int)
    .REQUIRED_ATTR(out_width, Int)
    .ATTR(border_type, String, "BORDER_CONSTANT")
    .ATTR(constant, Float, 0)
    .OP_END_FACTORY_REG(WarpPerspective)
}  // namespace ge

#endif  // GE_OP_WARP_PERSPECTIVE_OPS_H_
