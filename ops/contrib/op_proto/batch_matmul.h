/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: batch_matmul op proto h file
 * Author:
 * Create: 2020-6-11
 * Note:
 *
 * @file batch_matmul.h
 *
 * @brief Batch matrix multiplication
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_BATCHMATMUL_H
#define GE_OP_BATCHMATMULL_H

#include "graph/operator_reg.h"

namespace ge {
    /**
     * @par Inputs:
     *   2 inputs, including:
     * @li input_data_left: left matrix
     *       DataType: float16
     *       Format: ND
     * @li input_data_right: right matrix
     *        DataType: float16
     *        Format: ND
     *
     * @par Outputs:
     *   1 outputs, including:
     * @li output_data: output data
     *        DataType: float16
     *        Format: ND
     *
     * @par Attributes:
     * @li
     *
     * @attention Constraints:
     * @li
     */
    REG_OP(BatchMatmul)
        .INPUT(input_data_left, TensorType({DT_FLOAT16}))
        .INPUT(input_data_right, TensorType({DT_FLOAT16}))
        .OUTPUT(output_data, TensorType({DT_FLOAT16}))
        .OP_END_FACTORY_REG(BatchMatmul)
}

#endif
