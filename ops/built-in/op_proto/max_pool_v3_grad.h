/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
 */

/*!
 *\file max_pool_v3_grad.h
 *\brief
 */
#ifndef GE_OP_MAX_POOL_V3_GRAD_H
#define GE_OP_MAX_POOL_V3_GRAD_H

#include "graph/operator_reg.h"

namespace ge {

REG_OP(MaxPoolV3Grad)
    .INPUT(orig_input, TensorType::RealNumberType())
    .INPUT(orig_output, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .OUTPUT(out_grad, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mod, String, "CALCULATED")
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(data_format, String, "NCHW")
    .ATTR(global_pooling, Bool, false)
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolV3Grad)
}  // namespace ge

#endif  // GE_OP_MAX_POOL_V3_GRAD_H
