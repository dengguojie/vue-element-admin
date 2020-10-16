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
 * @file data_flow_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/batch_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/lookup_ops_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(Batch, BatchInfer)
{
    for (size_t i = 0; i < op.GetInputsSize(); ++i) {
        Shape out_shapes;
        if (ReplaceDim(op.GetInputDesc(i).GetShape(), 0, ge::UNKNOWN_DIM, out_shapes, op.GetName().c_str()) ==
            GRAPH_FAILED) {
            OP_LOGE(op.GetName().c_str(), "input param is error");
            return GRAPH_FAILED;
        }
        auto y_tensor_type = op.GetDynamicInputDesc("x_tensors", i).GetDataType();
        TensorDesc output_desc = op.GetDynamicOutputDesc("y_tensors", i);
        output_desc.SetShape(out_shapes);
        output_desc.SetDataType(y_tensor_type);
        op.UpdateDynamicOutputDesc("y_tensors", i, output_desc);
    }

    Shape scalar_shape;
    Scalar(scalar_shape);
    TensorDesc y_desc = op.GetOutputDesc("y_id");
    y_desc.SetShape(scalar_shape);
    y_desc.SetDataType(DT_INT64);
    op.UpdateOutputDesc("y_id", y_desc);

    std::vector<int64_t> dims = { ge::UNKNOWN_DIM, 3 };
    TensorDesc output_desc_batch_index = op.GetOutputDesc("y_index");
    output_desc_batch_index.SetShape(Shape(dims));
    output_desc_batch_index.SetDataType(DT_INT64);
    op.UpdateOutputDesc("y_index", output_desc_batch_index);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Batch, BatchInfer);

IMPLEMT_INFERFUNC(Unbatch, UnbatchInfer)
{
    Shape out_shape;
    auto x_tensor_tensor = op.get_input_desc_x_tensor();
    if (ReplaceDim(op.GetInputDesc(0).GetShape(), 0, ge::UNKNOWN_DIM, out_shape, op.GetName().c_str()) ==
        GRAPH_FAILED) {
        OP_LOGE(op.GetName().c_str(), "create y_tensors shape failed");
        return GRAPH_FAILED;
    }
    TensorDesc outputDesc = op.GetOutputDesc("y_tensor");
    outputDesc.SetShape(out_shape);
    outputDesc.SetDataType(x_tensor_tensor.GetDataType());
    if (op.UpdateOutputDesc("y_tensor", outputDesc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y_tensor desc failed");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Unbatch, UnbatchInfer);

IMPLEMT_INFERFUNC(UnbatchGrad, UnbatchGradInfer)
{
    auto x_input_tensor = op.get_input_desc_x_input();
    auto grad_tensor = op.get_input_desc_grad();
    auto grad_rank = grad_tensor.GetShape().GetDimNum();
    if (x_input_tensor.GetDataType() != grad_tensor.GetDataType()) {
        OP_LOGE(op.GetName().c_str(), "x_input's data type != grad's data type");
        return GRAPH_FAILED;
    }
    auto out_shape = UnknownShapeOfRank(grad_rank);
    TensorDesc outputDesc = op.GetOutputDesc("y_grad");
    outputDesc.SetShape(out_shape);
    outputDesc.SetDataType(x_input_tensor.GetDataType());
    if (op.UpdateOutputDesc("y_grad", outputDesc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y_grad desc failed");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(UnbatchGrad, UnbatchGradInfer);
}   // namespace ge
