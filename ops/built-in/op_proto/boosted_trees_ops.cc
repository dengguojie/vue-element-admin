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
 * @file boosted_trees_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/boosted_trees_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(BoostedTreesBucketize, BoostedTreesBucketizeInfer)
{
    int64_t num_features;
    if (op.GetAttr("num_features", num_features) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (num_features < 0) {
        OP_LOGE(op.GetName().c_str(), "input num_features must be >= 0.");
        return GRAPH_FAILED;
    }

    op.create_dynamic_output_y(num_features);
    for (int32_t i = 0; i < num_features; i++) {
        Shape value_shape;
        if (WithRank(op.GetInputDesc(i), 2, value_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
            OP_LOGE(op.GetName().c_str(), "each member in float_values list must be 2-D.");
            return GRAPH_FAILED;
        }

        Shape boundaries_shape;
        if (WithRank(op.GetInputDesc(i + num_features), 1, boundaries_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
            OP_LOGE(op.GetName().c_str(), "each member in bucket_boundaries list must be 1-D.");
            return GRAPH_FAILED;
        }

        int64_t unused_dim = 0;
        if (Merge(value_shape.GetDim(0), op.GetInputDesc(0).GetShape().GetDim(0), unused_dim)) {
            OP_LOGE(op.GetName().c_str(), "the dim(0) of each member in float_values list must be equal.");
            return GRAPH_FAILED;
        }

        int64_t y_dim0 = value_shape.GetDim(0);
        TensorDesc y_desc = op.GetDynamicOutputDesc("y", i);
        Shape y_shape({ y_dim0, 1 });
        y_desc.SetShape(y_shape);
        y_desc.SetDataType(DT_INT32);
        if (op.UpdateDynamicOutputDesc("y", i, y_desc) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BoostedTreesBucketize, BoostedTreesBucketizeInfer);
}  // namespace ge
