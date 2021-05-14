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
 * @file fasterrcnn_cropandresize.h
 *
 * @version 1.0
 */

#ifndef GE_OP_FASTERRCNNCROPANDRESIZE_H
#define GE_OP_FASTERRCNNCROPANDRESIZE_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Extracts crops from the input image tensor and resizes them.
 *
 * @par Inputs:
 * 2 inputs, including:
 * @li featuremap_gm: required. input feature map
 * DataType: float16
 * Format: NHWC
 * @li mapboxout_gm. required. input map box
 * DataType: float16
 * Format: ND
 *
 * @par Outputs:
 * 1 outputs, including:
 * @li cropandresize_gm. required. output
 * DataType: float16
 * Format: NHWC
 *
 *
 * @attention Constraints:
 * @li limits to two set of shapes:
 * {1, 40, 128, 1088}, {300, 4};
 * {1, 38, 64, 1024}, {100, 4};
 *
 *
 */

// namespace ge
REG_OP(FasterrcnnCropandresizeTik)
    .INPUT(featuremap_gm, TensorType({ DT_FLOAT16 }))
    .INPUT(mapboxout_gm, TensorType({ DT_FLOAT16 }))
    .OUTPUT(cropandresize_gm, TensorType({ DT_FLOAT16 }))
    .OP_END_FACTORY_REG(FasterrcnnCropandresizeTik)
}

#endif // GE_OP_FASTERRCNNCROPANDRESIZE_H
