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
 * @file ragged_array_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/ragged_array_ops.h"
#include <utility>
#include "graph/operator.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/util.h"

namespace ge {
IMPLEMT_INFERFUNC(RaggedGather, RaggedGatherInfer)
{
    int num_splits;
    int64_t PARAMS_RAGGED_RANK;
    if (op.GetAttr("PARAMS_RAGGED_RANK", PARAMS_RAGGED_RANK) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get attr PARAMS_RAGGED_RANK failed");
        return GRAPH_FAILED;
    }
    if (op.GetAttr("OUTPUT_RAGGED_RANK", num_splits) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get attr OUTPUT_RAGGED_RANK failed");
        return GRAPH_FAILED;
    }

    OP_LOGE(op.GetName().c_str(), "PARAMS_RAGGED_RANK:%ld", PARAMS_RAGGED_RANK);
    OP_LOGE(op.GetName().c_str(), "num_splits:%d", num_splits);

    Shape indices;
    if (WithRank(op.get_input_desc_indices(), num_splits - PARAMS_RAGGED_RANK + 1, indices, op.GetName().c_str()) !=
        GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input indices rank error.");
        return GRAPH_FAILED;
    }

    for (int64_t i = 0; i < PARAMS_RAGGED_RANK; ++i) {
        Shape indices;
        if (WithRank(op.GetDynamicInputDesc("params_nested_splits", i), 1, indices, op.GetName().c_str()) !=
            GRAPH_SUCCESS) {
            OP_LOGE(op.GetName().c_str(), "input params_nested_splits:%d rank must be 1.", i);
            return GRAPH_FAILED;
        }
    }

    Shape params_dense_values;
    if (WithRankAtLeast(op.get_input_desc_params_dense_values(), 1, params_dense_values, op.GetName().c_str()) !=
        GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input params_dense_values must be at least 1.");
        return GRAPH_FAILED;
    }

    DataType Tsplits_type;
    if (op.GetAttr("Tsplits", Tsplits_type) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get attr Tsplits error.");
    }
    DataType Tvalues_type = op.GetInputDesc("params_dense_values").GetDataType();
    std::vector<int64_t> unknow_dim_vec(1, UNKNOWN_DIM);
    for (int64_t i = 0; i < num_splits; ++i) {
        TensorDesc y_tensor = op.GetDynamicOutputDesc("output_nested_splits", i);
        y_tensor.SetDataType(Tsplits_type);
        y_tensor.SetShape(Shape(unknow_dim_vec));
        if (op.UpdateDynamicOutputDesc("output_nested_splits", i, y_tensor) != GRAPH_SUCCESS) {
            OP_LOGE(op.GetName().c_str(), "update y_%ld desc failed", i);
            return GRAPH_FAILED;
        }
    }

    Shape value(ge::UNKNOWN_SHAPE);
    Shape values(ge::UNKNOWN_SHAPE);
    if (SubShape(params_dense_values, 1, params_dense_values.GetDimNum(), 1, value, op.GetName().c_str()) !=
        GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get SubShape value error.");
        return GRAPH_FAILED;
    }
    if (Concatenate(Shape(unknow_dim_vec), value, values) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get Concatenate value error.");
        return GRAPH_FAILED;
    }

    TensorDesc outputDesc = op.GetOutputDesc("output_dense_values");
    outputDesc.SetDataType(Tvalues_type);
    outputDesc.SetShape(values);
    if (op.UpdateOutputDesc("output_dense_values", outputDesc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update output_dense_values desc failed");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RaggedGather, RaggedGatherInfer);

}  // namespace ge
