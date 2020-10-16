/* *
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file resource_variable_ops_shape_fns.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "resource_variable_ops_shape_fns.h"
#include "graph/types.h"
#include "op_log.h"
#include "common_shape_fns.h"

namespace ge {
graphStatus CreateAssignShapeFn(Operator &op)
{
    std::vector<ShapeAndType> shape_and_type;
    if (ValidateVariableResourceHandle(op, shape_and_type) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Validate variable resource handle failed.");
        return GRAPH_FAILED;
    }

    Shape shape = op.GetInputDesc(1).GetShape();
    Shape unused;
    if (Merge(shape_and_type[0].GetShape(), shape, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Merge input 0 and 1 shape failed.");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}
} // namespace ge
