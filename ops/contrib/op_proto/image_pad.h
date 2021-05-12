/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file image_pad.h
 *
 * @version 1.0
 */

#ifndef GE_OP_IMAGEPAD_H
#define GE_OP_IMAGEPAD_H

#include "graph/operator_reg.h"

/**
 * @brief H、W、C aixs pad constant value for image
 *
 * @par Inputs:
 * 1 inputs, including:
 * @li input_image: input image, a ND tensor of type is: float16
 *
 * @par Attributes:
 * @li paddings: A Tensor of type int32 or int64.
 * @li pad_value: constant value, required, dtype=int
 *
 * @par Outputs:
 * 1 outputs, including:
 * @li output_image: output image after pad , a ND tensor of type is float16
 *
 * @attention Constraints:
 * @li input or output image mem size <= 1GB
 */
namespace ge {
REG_OP(ImagePad)
    .INPUT(input_dict, TensorType({ DT_FLOAT16 }))   /* "First operand." */
    .OUTPUT(output_dict, TensorType({ DT_FLOAT16 })) /* "Result, has same element type as two inputs" */
    .REQUIRED_ATTR(paddings, ListListInt)
    .REQUIRED_ATTR(pad_value, Int)
    .OP_END_FACTORY_REG(ImagePad)
}

#endif // GE_OP_IMAGEPAD_H
