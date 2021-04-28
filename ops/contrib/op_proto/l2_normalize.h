/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: l2_normalize op proto h file
 * Author:
 * Create: 2020-6-11
 * Note:
 *
 * @file l2_normalize.h
 *
 * @brief Data normalization
 *
 * @version 1.0
 *
 */
#ifndef GE_OP_L2NORMALIZE_H
#define GE_OP_L2NORMALIZE_H
#include "graph/operator_reg.h"

namespace ge {
    /**
     *
     * @brief Data normalization
     *
     * @par Inputs:
     *   1 inputs, including:
     * @li input_data_gm : input data
     *        DataType: float16
     *        Format: NC1HWC0
     *
     * @par Outputs:
     *   1 outputs, including:
     * @li output_data_gm : normalized data
     *        DataType: float16
     *        Format: NC1HWC0
     *
     * @par Attributes:
     * @li eps: A minimum value replaces 0
     *
     * @attention Constraints:
     * @li input shape only support (1, 512, 1, 1)
     */
    REG_OP(L2Normalize)
        .INPUT(input_data_gm, TensorType({DT_FLOAT16}))
        .ATTR(eps, Float, 1e-10)
        .OUTPUT(output_data_gm, TensorType({DT_FLOAT16}))
        .OP_END_FACTORY_REG(L2Normalize)
}

#endif  // GE_OP_L2NORMALIZE_H
