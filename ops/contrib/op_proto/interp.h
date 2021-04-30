/**
* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
* Description: op_proto for interp
* Author: Huawei
* Create: 20200410
*
* @file interp.h
*
* @version 1.0
 */

#ifndef GE_OP_INTERP_H
#define GE_OP_INTERP_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief image interpolation using resize bilinear
*
* @par Inputs:
* 1 inputs, including:
* images: Image to be interpolated, a NC1HWC0 tensor of type is: float16, \n
*  float32
*
* @par Attributes:
* @li height: Height of the interpolated image, optional, dtype=int, \n
*  height < = 2048, default = 0
* @li width: Width of the interpolated image, optional, dtype=int, \n
*  width <= 2048, default = 0
* @li zoom_factor: Zoom factor of the interpolated image, An optional int, \n
*  default = 1
* @li shrink_factor: Shrink factor of the interpolated image, An optional int, \n
*  default = 1
* @li pad_beg: Pad front boundary of the interpolated image, An optional int, \n
*  default = 0, not support to setting and setting does not take effect.
* @li pad_end: Pad back boundary of the interpolated image,  An optional int, \n
*  default = 0, not support to setting and setting does not take effect.
*
* @par Outputs:
* 1 outputs, including:
* y: Image after interpolation , a NC1HWC0 tensor of type is float32
*
* @attention Constraints:
* @li input h, w <= 2048
* @li attribute priority is zoom_factor/shrink_factor > height/width
*/
REG_OP(Interp)
    .INPUT(images, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(height, Int, 0)
    .ATTR(width, Int, 0)
    .ATTR(zoom_factor, Int, 1)
    .ATTR(shrink_factor, Int, 1)
    .ATTR(pad_beg, Int, 0)
    .ATTR(pad_end, Int, 0)
    .OP_END_FACTORY_REG(Interp)
}
#endif // GE_OP_INTERP_H
