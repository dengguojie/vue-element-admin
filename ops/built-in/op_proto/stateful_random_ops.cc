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
 * @file stateful_random_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/stateful_random_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/util.h"

namespace ge {
IMPLEMT_INFERFUNC(NonDeterministicInts, NonDeterministicIntsInfer)
{
    Shape shape;
    Tensor shape_tensor;
    if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get shape_tensor error.");
        return GRAPH_FAILED;
    }
    if (MakeShapeFromShapeTensor(shape_tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get shape error.");
        return GRAPH_FAILED;
    }

    DataType dtype;
    if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get attr dtype error.");
        return GRAPH_FAILED;
    }
    TensorDesc outputDesc = op.GetOutputDesc("y");
    outputDesc.SetDataType(dtype);
    outputDesc.SetShape(shape);
    return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(NonDeterministicInts, NonDeterministicIntsInfer);

IMPLEMT_INFERFUNC(RngSkip, RngSkipInfer)
{
    Shape unused_shape;
    if (WithRank(op.GetInputDesc(1), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input unused_shape rank must be 0.");
        return GRAPH_FAILED;
    }
    Shape unused_shape1;
    if (WithRank(op.GetInputDesc(2), 0, unused_shape1, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input unused_shape1 rank must be 0.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RngSkip, RngSkipInfer);

IMPLEMT_INFERFUNC(StatefulRandomBinomial, StatefulRandomBinomialInfer)
{
    Shape unused;
    if (WithRankAtMost(op.GetInputDesc(3), 1, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input counts rank must be 1");
        return GRAPH_FAILED;
    }
    if (WithRankAtMost(op.GetInputDesc(4), 1, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input probs rank must be 1");
        return GRAPH_FAILED;
    }

    Shape shape;
    Tensor shape_tensor;
    if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get shape_tensor error.");
        return GRAPH_FAILED;
    }
    if (MakeShapeFromShapeTensor(shape_tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get shape error.");
        return GRAPH_FAILED;
    }

    TensorDesc outputDesc = op.GetOutputDesc("y");
    DataType output_type;
    if (op.GetAttr("dtype", output_type) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get attr output_dtype error.");
    }
    outputDesc.SetDataType(output_type);
    outputDesc.SetShape(shape);
    return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatefulRandomBinomial, StatefulRandomBinomialInfer);

IMPLEMT_INFERFUNC(StatefulStandardNormalV2, StatefulStandardNormalV2Infer)
{
    Shape unused;
    if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input algorithm rank must be 0");
        return GRAPH_FAILED;
    }

    Shape shape;
    Tensor shape_tensor;
    if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get shape_tensor error.");
        return GRAPH_FAILED;
    }
    if (MakeShapeFromShapeTensor(shape_tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get shape error.");
        return GRAPH_FAILED;
    }

    TensorDesc outputDesc = op.GetOutputDesc("y");
    outputDesc.SetDataType(DT_FLOAT);
    outputDesc.SetShape(shape);
    return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatefulStandardNormalV2, StatefulStandardNormalV2Infer);

IMPLEMT_INFERFUNC(StatefulTruncatedNormal, StatefulTruncatedNormalInfer)
{
    Shape unused;
    if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input algorithm rank must be 0");
        return GRAPH_FAILED;
    }

    Shape shape;
    Tensor shape_tensor;
    if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get shape_tensor error.");
        return GRAPH_FAILED;
    }
    if (MakeShapeFromShapeTensor(shape_tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get shape error.");
        return GRAPH_FAILED;
    }

    TensorDesc outputDesc = op.GetOutputDesc("y");
    outputDesc.SetDataType(DT_FLOAT);
    outputDesc.SetShape(shape);
    return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatefulTruncatedNormal, StatefulTruncatedNormalInfer);

IMPLEMT_INFERFUNC(StatefulUniform, StatefulUniformInfer)
{
    Shape unused;
    if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input algorithm rank must be 0");
        return GRAPH_FAILED;
    }

    Shape shape;
    Tensor shape_tensor;
    if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get shape_tensor error.");
        return GRAPH_FAILED;
    }
    if (MakeShapeFromShapeTensor(shape_tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get shape error.");
        return GRAPH_FAILED;
    }

    TensorDesc outputDesc = op.GetOutputDesc("y");
    outputDesc.SetDataType(DT_FLOAT);
    outputDesc.SetShape(shape);
    return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatefulUniform, StatefulUniformInfer);

IMPLEMT_INFERFUNC(StatefulUniformFullInt, StatefulUniformFullIntInfer)
{
    Shape unused;
    if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input algorithm rank must be 0");
        return GRAPH_FAILED;
    }

    Shape shape;
    Tensor shape_tensor;
    if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get shape_tensor error.");
        return GRAPH_FAILED;
    }
    if (MakeShapeFromShapeTensor(shape_tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get shape error.");
        return GRAPH_FAILED;
    }

    TensorDesc outputDesc = op.GetOutputDesc("y");
    outputDesc.SetDataType(DT_UINT64);
    outputDesc.SetShape(shape);
    return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatefulUniformFullInt, StatefulUniformFullIntInfer);

IMPLEMT_INFERFUNC(StatefulUniformInt, StatefulUniformIntInfer)
{
    Shape unused;
    if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input algorithm rank must be 0");
        return GRAPH_FAILED;
    }
    if (WithRank(op.GetInputDesc(3), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input minval rank must be 0");
        return GRAPH_FAILED;
    }
    if (WithRank(op.GetInputDesc(4), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input maxval rank must be 0");
        return GRAPH_FAILED;
    }

    Shape shape;
    Tensor shape_tensor;
    if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get shape_tensor error.");
        return GRAPH_FAILED;
    }
    if (MakeShapeFromShapeTensor(shape_tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get shape error.");
        return GRAPH_FAILED;
    }

    TensorDesc outputDesc = op.GetOutputDesc("y");
    outputDesc.SetDataType(DT_INT64);
    outputDesc.SetShape(shape);
    return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatefulUniformInt, StatefulUniformIntInfer);

}  // namespace ge
