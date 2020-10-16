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
 * @file linalg_ops_shape_fns.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_LINALG_OPS_SHAPE_FNS_H
#define GE_LINALG_OPS_SHAPE_FNS_H

#include "graph/tensor.h"

namespace ge {

/**
 * Generate a square matrix's Shape
 * @param tensor Input tensor
 * @param out Output Shape
 * @return status whether this operation success
 */
graphStatus MakeBatchSquareMatrix(const TensorDesc &tensor, Shape &out, const char* op_name);
/**
 * Solving linear equations from matrices common shape func
 * @param tensor1 first input tensor
 * @param tensor2 second input tensor
 * @param square whether matrix is square
 * @param out Output Shape
 * @return status whether this operation success
 */
graphStatus MatrixSolve(const TensorDesc &tensor1, const TensorDesc &tensor2,
                        bool square, Shape &out, const char* op_name);

}  // namespace ge

#endif  // GE_LINALG_OPS_SHAPE_FNS_H
