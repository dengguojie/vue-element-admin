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
 * @file array_ops_shape_fns.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "array_ops_shape_fns.h"
#include "graph/types.h"
#include "op_log.h"
#include "error_util.h"
#include "common_shape_fns.h"

namespace ge {
static graphStatus PadKnown(Operator &op, const Tensor &paddings_tensor,
                            const int64_t input_dim_num)
{
    TensorDesc paddings_tensor_desc = paddings_tensor.GetTensorDesc();
    DataType data_type = paddings_tensor_desc.GetDataType();
    std::vector<int64_t> data;
    // every dim has 2 element
    int64_t element_num = input_dim_num * 2;
    data.reserve(element_num);
    if (data_type == DT_INT32) {
        const int32_t *paddings_data = reinterpret_cast<const int32_t *>(paddings_tensor.GetData());
        for (int64_t i = 0; i < element_num; ++i) {
            data.push_back(static_cast<int64_t>(paddings_data[i]));
        }
    } else if (data_type == DT_INT64) {
        const int64_t *paddings_data = reinterpret_cast<const int64_t *>(paddings_tensor.GetData());
        for (int64_t i = 0; i < element_num; ++i) {
            data.push_back(paddings_data[i]);
        }
    } else {
        string err_msg = Strcat("paddings data type invalid, ", "should be DT_INT32 or DT_INT64");
        InferShapeOtherErrReport(op.GetName(), err_msg);
        OP_LOGE(op.GetName().c_str(),"%s", err_msg.c_str());
        return GRAPH_FAILED;
    }
    auto dims = op.GetInputDesc(0).GetShape().GetDims();
    std::vector<int64_t> output_dims(input_dim_num, UNKNOWN_DIM);
    if (dims != UNKNOWN_SHAPE) {
        output_dims.assign(dims.begin(), dims.end());
    }
    for (size_t i = 0; i < data.size(); i += 2) {
        if ((data[i] < 0) || (data[i + 1] < 0)) {
            std::string err_msg = Strcat("paddings", DebugString(data), " must be non-negative");
            InferShapeOtherErrReport(op.GetName(), err_msg);
            OP_LOGE(op.GetName().c_str(), "%s", err_msg.c_str());
            return GRAPH_FAILED;
        }
        graphStatus status = Add(output_dims[i / 2], data[i] + data[i + 1], output_dims[i / 2]);
        if (status != GRAPH_SUCCESS) {
            std::string err_msg = Strcat("the sum input[0] shape", DebugString(dims), " and input[1] value",
                DebugString(data), " must be non-negative");
            OP_LOGE(op.GetName().c_str(), "%s", err_msg.c_str());
            return GRAPH_FAILED;
        }
    }
    auto output_desc = op.GetOutputDesc("y");
    output_desc.SetShape(Shape(output_dims));

    return op.UpdateOutputDesc("y", output_desc);
}

graphStatus PadShapeFn(Operator &op)
{
    Shape paddings;
    int64_t input_dim_num;
    graphStatus status = WithRank(op.GetInputDesc(1), 2, paddings, op.GetName().c_str());
    if (status != GRAPH_SUCCESS) {
        ShapeErrReport(1, op.GetName(), DebugString(op.GetInputDesc(1).GetShape().GetDims()), "2D");
        return GRAPH_FAILED;
    }
    status = WithValue(paddings.GetDim(1), 2, input_dim_num, op.GetName().c_str());
    if (status != GRAPH_SUCCESS) {
        ShapeErrReport(1, op.GetName(), DebugString(op.GetInputDesc(1).GetShape().GetDims()), Strcat(2, " of dim[1]"));
        return GRAPH_FAILED;
    }
    Shape input;
    int64_t dim0 = paddings.GetDim(0);
    if (dim0 != UNKNOWN_DIM) {
        status = WithRank(op.GetInputDesc(0), dim0, input, op.GetName().c_str());
        if (status != GRAPH_SUCCESS) {
            ShapeErrReport(0, op.GetName(), DebugString(op.GetInputDesc(0).GetShape().GetDims()), Strcat(dim0, "D"));
            return GRAPH_FAILED;
        }
    } else if (op.GetInputDesc(0).GetShape().GetDim(0) != 0) {
        status = WithValue(dim0, op.GetInputDesc(0).GetShape().GetDimNum(), input_dim_num, op.GetName().c_str());
        if (status != GRAPH_SUCCESS) {
            ShapeErrReport(0, op.GetName(), DebugString(op.GetInputDesc(0).GetShape().GetDims()), Strcat(dim0, "D"));
            return GRAPH_FAILED;
        }
    }
    TensorDesc output_desc = op.GetOutputDesc("y");
    Tensor paddings_tensor;
    status = op.GetInputConstData("paddings", paddings_tensor);
    if (status != GRAPH_SUCCESS) {
        if (dim0 != UNKNOWN_DIM) {
            std::vector<int64_t> output_shape(dim0, UNKNOWN_DIM);
            output_desc.SetShape(Shape(output_shape));
        } else {
            output_desc.SetShape(Shape(UNKNOWN_SHAPE));
        }
        return op.UpdateOutputDesc("y", output_desc);
    }
    input_dim_num = paddings_tensor.GetTensorDesc().GetShape().GetDim(0);
    status = WithRank(op.GetInputDesc(0), input_dim_num, input, op.GetName().c_str());
    if(status == GRAPH_FAILED) {
        OP_LOGE(op.GetName().c_str(), "WithRank fail");
        return GRAPH_FAILED;
    }
    status = WithValue(dim0, input_dim_num, dim0, op.GetName().c_str());
    if(status == GRAPH_FAILED) {
        OP_LOGE(op.GetName().c_str(), "WithValue fail");
        return GRAPH_FAILED;
    }
    return PadKnown(op, paddings_tensor, input_dim_num);
}

static graphStatus CalcPadGradOutDims(const Shape &input_shape, const Tensor &paddings_tensor,
    std::vector<int64_t> &output_dims, const char* op_name)
{
    graphStatus status;
    size_t input_rank = input_shape.GetDimNum();
    if (output_dims.size() < input_rank) {
        return GRAPH_FAILED;
    }
    DataType padding_type = paddings_tensor.GetTensorDesc().GetDataType();
    if (padding_type == DT_INT32) {
        const int32_t *paddings_data = reinterpret_cast<const int32_t *>(paddings_tensor.GetData());
        for (size_t i = 0; i < input_rank; ++i) {
            const int64_t pad0 = static_cast<int64_t>(paddings_data[2 * i]);
            const int64_t pad1 = static_cast<int64_t>(paddings_data[(2 * i) + 1]);
            if ((pad0 < 0) || (pad1 < 0)) {
                OP_LOGE(op_name, "Paddings must be non-negative, pad0= %lld, pad1=%lld.", pad0, pad1);
                return GRAPH_FAILED;
            }
            status = Subtract(input_shape.GetDim(i), pad0 + pad1, output_dims[i], op_name);
            if (status != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
    } else if (padding_type == DT_INT64) {
        const int64_t *paddings_data = reinterpret_cast<const int64_t *>(paddings_tensor.GetData());
        for (size_t i = 0; i < input_rank; ++i) {
            const int64_t pad0 = paddings_data[2 * i];
            const int64_t pad1 = paddings_data[(2 * i) + 1];
            if ((pad0 < 0) || (pad1 < 0)) {
                OP_LOGE(op_name, "Paddings must be non-negative, pad0=%lld, pad1=%lld.", pad0, pad1);
                return GRAPH_FAILED;
            }
            status = Subtract(input_shape.GetDim(i), pad0 + pad1, output_dims[i], op_name);
            if (status != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
    } else {
        OP_LOGE(op_name, "Data type invalid, should be DT_INT32 or DT_INT64");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

graphStatus PadGradShapeFn(Operator &op)
{
    Shape paddings;
    graphStatus status = WithRank(op.GetInputDesc(1), 2, paddings, op.GetName().c_str());
    if (status != GRAPH_SUCCESS) {
        ShapeErrReport(1, op.GetName(), DebugString(op.GetInputDesc(1).GetShape().GetDims()), "2D");
        return GRAPH_FAILED;
    }
    int64_t input_rank = paddings.GetDim(0);
    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetDataType(op.GetInputDesc(0).GetDataType());
    if (input_rank == UNKNOWN_DIM) {
        OP_LOGE(op.GetName().c_str(), "paddings inputShape of 0 dims is unknown, set out shape unknown.");
        output_desc.SetShape(Shape(UNKNOWN_SHAPE));
        return op.UpdateOutputDesc("y", output_desc);
    }

    Shape input_shape;
    if (WithRank(op.GetInputDesc(0), input_rank, input_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        ShapeErrReport(0, op.GetName(), DebugString(op.GetInputDesc(0).GetShape().GetDims()), Strcat(input_rank));
        return GRAPH_FAILED;
    }

    Shape check_shape({ input_rank, 2 });
    if (Merge(paddings, check_shape, paddings, op.GetName().c_str())) {
        string err_msg = Strcat("merge 1th input shape", DebugString(paddings.GetDims()), " and shape",
            DebugString(check_shape.GetDims()), " failed");
        InferShapeOtherErrReport(op.GetName(), err_msg);
        OP_LOGE(op.GetName().c_str(), "Input dimension mismatch, inputRank=%lld.", input_rank);
        return GRAPH_FAILED;
    }

    Tensor paddings_tensor;
    if (op.GetInputConstData("paddings", paddings_tensor) != GRAPH_SUCCESS) {
        std::vector<int64_t> unknow_dim_vec(input_rank, UNKNOWN_DIM);
        OP_LOGE(op.GetName().c_str(), "Get paddings input tensor fail, set outPut shape unknown.");
        output_desc.SetShape(Shape(unknow_dim_vec));
        return op.UpdateOutputDesc("y", output_desc);
    }

    std::vector<int64_t> output_dims(input_rank);
    auto result = CalcPadGradOutDims(input_shape, paddings_tensor, output_dims, op.GetName().c_str());
    if (result != GRAPH_SUCCESS) {
        string err_msg = Strcat("calculate out dims failed,", "please check the validity of input and attribute");
        InferShapeOtherErrReport(op.GetName(), err_msg);
        OP_LOGE(op.GetName().c_str(), "Calculation PadGrad out dimensions failed.");
        return GRAPH_FAILED;
    }
    output_desc.SetShape(Shape(output_dims));
    return op.UpdateOutputDesc("y", output_desc);
}
} // namespace ge
