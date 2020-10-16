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
 * @file stateless_random_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/stateless_random_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(StatelessMultinomial, StatelessMultinomialInfer)
{
    auto logitsTensor = op.get_input_desc_logits();
    auto num_samplesTensor = op.get_input_desc_num_samples();
    auto seedTensor = op.get_input_desc_seed();

    Shape seed;
    if (WithRank(seedTensor, 1, seed, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input seed rank must be 1.");
        return GRAPH_FAILED;
    }
    if (seed.GetDim(0) != 2) {
        OP_LOGE(op.GetName().c_str(), "The value of dim-0 for seed must be 2, real value is %ld.", seed.GetDim(0));
        return GRAPH_FAILED;
    }

    Shape logits_shape;
    Shape unused;
    if (WithRank(logitsTensor, 2, logits_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input logits_shape rank must be 2.");
        return GRAPH_FAILED;
    }
    if (WithRank(num_samplesTensor, 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input num_samples rank must be 0.");
        return GRAPH_FAILED;
    }

    DataType dtype = op.get_input_desc_num_samples().GetDataType();
    if (dtype != DT_INT32) {
        OP_LOGE(op.GetName().c_str(), "input num_samples must be DT_INT32.");
        return GRAPH_FAILED;
    }

    Tensor numSamplesTensor;
    if (op.GetInputConstData("num_samples", numSamplesTensor) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "GetInputConstData error.");
        return GRAPH_FAILED;
    }
    int64_t numSamples;
    if (MakeDimForScalarInput(numSamplesTensor, numSamples, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input numSamples MakeDimForScalarInput error.");
        return GRAPH_FAILED;
    }

    Shape shape;
    (void)Matrix(logits_shape.GetDim(0), numSamples, shape);

    TensorDesc outputDesc = op.GetOutputDesc("y");
    DataType output_type;
    if (op.GetAttr("output_dtype", output_type) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Get attr output_dtype error.");
    }

    outputDesc.SetDataType(output_type);
    outputDesc.SetShape(shape);
    return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatelessMultinomial, StatelessMultinomialInfer);

IMPLEMT_INFERFUNC(StatelessRandomUniformInt, StatelessRandomUniformIntInfer)
{
    Shape unused;
    if (WithRank(op.GetInputDesc(2), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input minval rank must be 0");
        return GRAPH_FAILED;
    }
    if (WithRank(op.GetInputDesc(3), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input maxval rank must be 0");
        return GRAPH_FAILED;
    }

    Shape seed;
    if (WithRank(op.GetInputDesc(1), 1, seed, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input seed rank must be 1.");
        return GRAPH_FAILED;
    }
    if (seed.GetDim(0) != 2) {
        OP_LOGE(op.GetName().c_str(), "The value of dim-0 for seed must be 2, real value is %ld.", seed.GetDim(0));
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
    DataType output_type = op.GetInputDesc("minval").GetDataType();
    outputDesc.SetDataType(output_type);
    outputDesc.SetShape(shape);
    return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatelessRandomUniformInt, StatelessRandomUniformIntInfer);

}  // namespace ge
