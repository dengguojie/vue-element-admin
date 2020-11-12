/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
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
 *@file topk_cust.h
 *
 *@version 1.0
 *
 */

#ifndef GE_OP_UNIQUE_CUST_H
#define GE_OP_UNIQUE_CUST_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * *@brief Cast a tensor form src data type to dst data type.
 *
 * *@par Inputs:
 * *One input:
 * *x:A Tensor. Must be one of the following types: bool, float16, float, int8, int32, uint32, uint8,
 *    int64, uint64, int16, uint16, double, complex64, complex128, qint8, quint8, qint16, quint16, qint32.
 *
 *    *@par Attributes:
 *    *dst_type: An required attribute of type int32, specifying the dst data type.
 *
 *    *@par Outputs:
 *    *y:A Tensor. Has the same type as x.
 *    */
REG_OP(UniqueCust)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(idx, TensorType({DT_INT32, DT_INT64}))
    .ATTR(out_idx, Type, DT_INT32)
    .OP_END_FACTORY_REG(UniqueCust)
}

#endif // GE_OP_UNIQUE_CUST_H
