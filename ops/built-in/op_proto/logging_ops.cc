/* *
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http:// www.apache.org/licenses/LICENSE-2.0
 *
 * @file logging_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/logging_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(Timestamp, TimestampInfer)
{
    TensorDesc yDesc = op.GetOutputDesc("y");
    Shape scalarShape;
    (void)Scalar(scalarShape);
    yDesc.SetDataType(DT_DOUBLE);
    yDesc.SetShape(scalarShape);
    return op.UpdateOutputDesc("y", yDesc);
}

INFER_FUNC_REG(Timestamp, TimestampInfer);

IMPLEMT_INFERFUNC(Assert, AssertInfer)
{
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Assert, AssertInfer);

IMPLEMT_INFERFUNC(Print, PrintInfer)
{
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Print, PrintInfer);

IMPLEMT_INFERFUNC(PrintV2, PrintV2Infer)
{
    Shape unused;
    if (WithRank(op.GetInputDesc(0), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input x must be a scalar");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(PrintV2, PrintV2Infer);

}  // namespace ge
