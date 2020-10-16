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
 * @file ctc_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/ctc_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/util.h"

namespace ge {
IMPLEMT_INFERFUNC(CTCLoss, CTCLossInfer)
{
    Shape inputs;
    Shape labels_indices;
    Shape labels_values;
    Shape sequence_length;
    if (WithRank(op.GetInputDesc(0), 3, inputs, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input inputs rank must be 3");
        return GRAPH_FAILED;
    }
    if (WithRank(op.GetInputDesc(1), 2, labels_indices, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input labels_indices rank must be 2");
        return GRAPH_FAILED;
    }
    if (WithRank(op.GetInputDesc(2), 1, labels_values, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input labels_values rank must be 1");
        return GRAPH_FAILED;
    }
    if (WithRank(op.GetInputDesc(3), 1, sequence_length, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input sequence_length rank must be 1");
        return GRAPH_FAILED;
    }

    int64_t dim1 = labels_indices.GetDim(0);
    int64_t dim2 = labels_values.GetDim(0);
    int64_t unused = 0;
    if (Merge(dim1, dim2, unused) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Merge labels_indices and labels_values failed.");
        return GRAPH_FAILED;
    }
    int64_t dim3 = inputs.GetDim(1);
    int64_t dim4 = sequence_length.GetDim(0);
    int64_t batch_size = 0;
    if (Merge(dim3, dim4, batch_size) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Merge inputs and sequence_length failed.");
        return GRAPH_FAILED;
    }
    inputs.SetDim(1, batch_size);

    DataType type = op.GetInputDesc("inputs").GetDataType();
    TensorDesc loss_desc = op.GetOutputDesc("loss");
    loss_desc.SetShape(Shape({ batch_size }));
    loss_desc.SetDataType(type);
    if (op.UpdateOutputDesc("loss", loss_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "fail to update output loss.");
        return GRAPH_FAILED;
    }
    TensorDesc gradient_desc = op.GetOutputDesc("gradient");
    gradient_desc.SetShape(Shape(inputs));
    gradient_desc.SetDataType(type);
    if (op.UpdateOutputDesc("gradient", gradient_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "fail to update output gradient.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CTCLoss, CTCLossInfer);

IMPLEMT_INFERFUNC(CTCGreedyDecoder, CTCGreedyDecoderInfer)
{
    Shape inputs_shape;
    auto inputs_desc = op.GetInputDesc(0);
    if (WithRank(inputs_desc, 3, inputs_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input inputs rank must be 3, got rank %lld", inputs_desc.GetShape().GetDimNum());
        return GRAPH_FAILED;
    }

    Shape sequence_length_shape;
    auto sequence_length_desc = op.GetInputDesc(1);
    if (WithRank(sequence_length_desc, 1, sequence_length_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input sequence_length rank must be 1, got rank %lld",
            sequence_length_desc.GetShape().GetDimNum());
        return GRAPH_FAILED;
    }

    int64_t batch_size;
    if (Merge(inputs_shape.GetDim(1), sequence_length_shape.GetDim(0), batch_size) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "merge inputs dim 1 value %lld and sequence_length dim 0 value %lld faild",
            inputs_shape.GetDim(1), sequence_length_shape.GetDim(0));
        return GRAPH_FAILED;
    }

    auto total_decoded_outputs = UNKNOWN_DIM;

    auto decoded_indices_desc = op.GetOutputDesc("decoded_indices");
    decoded_indices_desc.SetShape(Shape({ total_decoded_outputs, 2 }));
    decoded_indices_desc.SetDataType(DT_INT64);
    if (op.UpdateOutputDesc("decoded_indices", decoded_indices_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "failed to update output decoded_indices");
        return GRAPH_FAILED;
    }

    auto decoded_values_desc = op.GetOutputDesc("decoded_values");
    decoded_values_desc.SetShape(Shape({ total_decoded_outputs }));
    decoded_values_desc.SetDataType(DT_INT64);
    if (op.UpdateOutputDesc("decoded_values", decoded_values_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "failed to update output decoded_values");
        return GRAPH_FAILED;
    }

    auto decoded_shape_desc = op.GetOutputDesc("decoded_shape");
    decoded_shape_desc.SetShape(Shape({ 2 }));
    decoded_shape_desc.SetDataType(DT_INT64);
    if (op.UpdateOutputDesc("decoded_shape", decoded_shape_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "failed to update output decoded_shape");
        return GRAPH_FAILED;
    }

    auto log_probability_desc = op.GetOutputDesc("log_probability");
    log_probability_desc.SetShape(Shape({ batch_size, 1 }));
    log_probability_desc.SetDataType(inputs_desc.GetDataType());
    if (op.UpdateOutputDesc("log_probability", log_probability_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "failed to update output log_probability");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CTCGreedyDecoder, CTCGreedyDecoderInfer);

IMPLEMT_INFERFUNC(CTCBeamSearchDecoder, CTCBeamSearchDecoderInfer)
{
    Shape inputs_shape;
    auto inputs_desc = op.GetInputDesc(0);
    if (WithRank(inputs_desc, 3, inputs_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input inputs rank must be 3, got rank %lld", inputs_desc.GetShape().GetDimNum());
        return GRAPH_FAILED;
    }

    Shape sequence_length_shape;
    auto sequence_length_desc = op.GetInputDesc(1);
    if (WithRank(sequence_length_desc, 1, sequence_length_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input sequence_length rank must be 1, got rank %lld",
            sequence_length_desc.GetShape().GetDimNum());
        return GRAPH_FAILED;
    }

    int64_t batch_size;
    if (Merge(inputs_shape.GetDim(1), sequence_length_shape.GetDim(0), batch_size) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "merge inputs dim 1 value %lld and sequence_length dim 0 value %lld faild",
            inputs_shape.GetDim(1), sequence_length_shape.GetDim(0));
        return GRAPH_FAILED;
    }

    int32_t top_paths;
    if (op.GetAttr("top_paths", top_paths) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "failed to get attr top_paths");
        return GRAPH_FAILED;
    }

    for (int i = 0; i < top_paths; ++i) {
        auto temp_desc = op.GetDynamicOutputDesc("decoded_indices", i);
        temp_desc.SetShape(Shape({ UNKNOWN_DIM, 2 }));
        temp_desc.SetDataType(DT_INT64);
        if (op.UpdateDynamicOutputDesc("decoded_indices", i, temp_desc) != GRAPH_SUCCESS) {
            OP_LOGE(op.GetName().c_str(), "failed to update dynamic output decoded_indices, id %lld", i);
            return GRAPH_FAILED;
        }
    }

    for (int i = 0; i < top_paths; ++i) {
        auto temp_desc = op.GetDynamicOutputDesc("decoded_values", i);
        temp_desc.SetShape(Shape({ UNKNOWN_DIM }));
        temp_desc.SetDataType(DT_INT64);
        if (op.UpdateDynamicOutputDesc("decoded_values", i, temp_desc) != GRAPH_SUCCESS) {
            OP_LOGE(op.GetName().c_str(), "failed to update dynamic output decoded_indices, id %lld", i);
            return GRAPH_FAILED;
        }
    }

    for (int i = 0; i < top_paths; ++i) {
        auto temp_desc = op.GetDynamicOutputDesc("decoded_shape", i);
        temp_desc.SetShape(Shape({ 2 }));
        temp_desc.SetDataType(DT_INT64);
        if (op.UpdateDynamicOutputDesc("decoded_shape", i, temp_desc) != GRAPH_SUCCESS) {
            OP_LOGE(op.GetName().c_str(), "failed to update dynamic output decoded_indices, id %lld", i);
            return GRAPH_FAILED;
        }
    }

    auto log_probability_desc = op.GetOutputDesc("log_probability");
    log_probability_desc.SetShape(Shape({ batch_size, top_paths }));
    log_probability_desc.SetDataType(inputs_desc.GetDataType());
    if (op.UpdateOutputDesc("log_probability", log_probability_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "faild to update output log_probability");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CTCBeamSearchDecoder, CTCBeamSearchDecoderInfer);
} // namespace ge