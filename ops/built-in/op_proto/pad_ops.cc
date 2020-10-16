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
 * @file pad_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include "inc/pad_ops.h"
#include <vector>
#include <string.h>
#include "op_log.h"
#include "util/util.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"
namespace ge {
// ----------------PadD Op Begin-------------------
static graphStatus PadDInferShapeAndType(ge::Operator &op, std::vector<std::vector<int64_t>> &paddings)
{
    Shape shape_x = op.GetInputDesc("x").GetShape();

    // adapter net
    vector<int64_t> shape;
    int64_t dim_cur = 0;
    if (shape_x.GetDimNum() != paddings.size()) {
        OpsInputShapeErrReport(op.GetName(), "Paddings and shape should be the same length", "x",
            Strcat(shape_x.GetDimNum()));
        OP_LOGE(op.GetName().c_str(), "Paddings and shape"
            "are not the same length.");
        return GRAPH_FAILED;
    }
    for (size_t dim = 0; dim < shape_x.GetDimNum(); dim++) {
        if (paddings[dim].size() != 2) {
            OpsInputShapeErrReport(op.GetName(), "Paddings's shape should be in the form of (n,2)", "paddings",
                Strcat(paddings[dim].size()));
            OP_LOGE(op.GetName().c_str(), "Paddings's shape"
                "is not in the form of (n,2)");
            return GRAPH_FAILED;
        }
    }
    for (size_t dim = 0; dim < shape_x.GetDimNum(); dim++) {
        dim_cur = shape_x.GetDim(dim) + paddings[dim][0] + paddings[dim][1];
        shape.push_back(dim_cur);
    }

    DataType input_dtype = op.GetInputDesc("x").GetDataType();
    Shape out_shape(shape);
    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    tensordesc_output.SetShape(out_shape);
    tensordesc_output.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", tensordesc_output);
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(PadDInferShape)
{
    std::vector<std::vector<int64_t>> paddings;
    if (ge::GRAPH_SUCCESS != op.GetAttr("paddings", paddings)) {
        OpsGetAttrErrReport(op.GetName(), "paddings");
        return GRAPH_FAILED;
    }

    return PadDInferShapeAndType(op, paddings);
}

COMMON_INFER_FUNC_REG(PadD, PadDInferShape);
// ----------------PadD Op End-------------------

// ----------------Pad Op Begin-------------------
static graphStatus PadInferShapeAndType(ge::Operator &op, std::vector<int64_t> &paddings)
{
    Shape shape_x = op.GetInputDesc("x").GetShape();

    // adapter net
    vector<int64_t> shape;
    int64_t dim_cur = 0;
    if (shape_x.GetDimNum() * 2 != paddings.size()) {
        return GRAPH_FAILED;
    }

    for (size_t dim = 0; dim < shape_x.GetDimNum(); dim++) {
        dim_cur = shape_x.GetDim(dim) + paddings[dim * 2] + paddings[dim * 2 + 1];
        shape.push_back(dim_cur);
    }

    for (size_t dim = 0; dim < shape_x.GetDimNum(); dim++) {
        if (shape_x.GetDim(dim) == UNKNOWN_DIM) {
            shape[dim] = UNKNOWN_DIM;
        }
    }

    DataType input_dtype = op.GetInputDesc("x").GetDataType();
    Shape out_shape(shape);
    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    tensordesc_output.SetShape(out_shape);
    tensordesc_output.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", tensordesc_output);
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(PadInferShape)
{
    Tensor paddings_tensor;
    if (ge::GRAPH_SUCCESS != op.GetInputConstData("paddings", paddings_tensor)) {
        Shape shape_x = op.GetInputDesc("x").GetShape();
        vector<int64_t> shape;
        for (size_t dim = 0; dim < shape_x.GetDimNum(); dim++) {
            shape.push_back(UNKNOWN_DIM);
        }
        DataType input_dtype = op.GetInputDesc("x").GetDataType();
        TensorDesc tensordesc_output = op.GetOutputDesc("y");
        Shape out_shape(shape);
        tensordesc_output.SetShape(out_shape);
        tensordesc_output.SetDataType(input_dtype);
        (void)op.UpdateOutputDesc("y", tensordesc_output);
        return GRAPH_SUCCESS;
    }
    DataType dtype = op.GetInputDesc("paddings").GetDataType();

    std::vector<int64_t> paddings;
    if (!GetConstValue(op, paddings_tensor, dtype, paddings)) {
        OP_LOGE(op.GetName().c_str(), "Get Const Value failed ");
        return GRAPH_FAILED;
    }

    return PadInferShapeAndType(op, paddings);
}

COMMON_INFER_FUNC_REG(Pad, PadInferShape);
// ----------------Pad Op End-------------------

// ----------------Fill Op Begin-------------------
template <typename T> static void CaclDims(const Tensor &data, std::vector<int64_t> &vec_dim)
{
    int32_t size = data.GetSize() / sizeof(T);
    for (int32_t i = 0; i < size; i++) {
        T dim = *((T *)data.GetData() + i);
        if (dim != 0) {
            vec_dim.push_back(dim);
        } else {
            vec_dim.clear();
            break;
        }
    }
}

IMPLEMT_INFERFUNC(Fill, FillInferShape)
{
    Tensor data;
    std::vector<int64_t> vec_dim;
    TensorDesc td = op.GetOutputDesc("y");
    if (op.GetInputConstData("dims", data) != GRAPH_SUCCESS) {
        OP_LOGW(op.GetName().c_str(), "Get constValue failed of [dims]");
        auto shape = op.GetInputDesc("dims").GetShape();
        int64_t dim_value;
        dim_value = shape.GetDim(0);
        for (int64_t m = 0; m < dim_value; m++) {
            vec_dim.push_back(-1);
        }
        td.SetShape(Shape(vec_dim));
        td.SetDataType(op.GetInputDesc("value").GetDataType());
        (void)op.UpdateOutputDesc("y", td);
        return GRAPH_SUCCESS;
    } else {
        op.GetInputConstData("dims", data);
        DataType data_type = data.GetTensorDesc().GetDataType();
        std::vector<int64_t> vec_dim;
        if (data_type == DT_INT32) {
            CaclDims<int32_t>(data, vec_dim);
        } else if (data_type == DT_INT64) {
            CaclDims<int64_t>(data, vec_dim);
        } else {
            GeInfershapeErrReport(op.GetName(), op.GetOpType(), "const dtype", "it must DT_INT32 or DT_INT64");
            OP_LOGE(op.GetName().c_str(), "Get constValue failed of [dims], the dtype must DT_INT32 or DT_INT64");
            return GRAPH_PARAM_INVALID;
        }
        td.SetShape(Shape(vec_dim));
        td.SetDataType(op.GetInputDesc("value").GetDataType());
        (void)op.UpdateOutputDesc("y", td);
        return GRAPH_SUCCESS;
    }
}

INFER_FUNC_REG(Fill, FillInferShape);
// ----------------Fill Op End-------------------

// ----------------FillD Op Begin-------------------
IMPLEMT_INFERFUNC(FillD, FillDInferShape)
{
    std::vector<int64_t> vec_dim;
    if (ge::GRAPH_SUCCESS != op.GetAttr("dims", vec_dim)) {
        OpsGetAttrErrReport(op.GetName(), "dims");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr failed of FillD!");
        return GRAPH_FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "start infershape");

    if (vec_dim.size() < DIM_SIZE1 || vec_dim.size() > DIM_SIZE8) {
        OpsInputShapeDimErrReport(op.GetName(), "dims", Strcat(DIM_SIZE8), Strcat(DIM_SIZE1), Strcat(vec_dim.size()));
        OP_LOGE(op.GetName().c_str(), "dims must be between 1 and 8.");
        return GRAPH_FAILED;
    }

    TensorDesc td = op.GetOutputDesc("y");
    td.SetShape(Shape(vec_dim));
    td.SetDataType(op.GetInputDesc("value").GetDataType());

    std::vector<std::pair<int64_t,int64_t>> range;
    auto status = op.GetInputDesc("value").GetShapeRange(range);
    if (status != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
    }
    td.SetShapeRange(range);
    
    (void)op.UpdateOutputDesc("y", td);
    OP_LOGI(op.GetName().c_str(), "infershape success");
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FillD, FillDInferShape);
// ----------------FillD Op End-------------------

// -------------------BroadcastTo-----------------------
IMPLEMT_INFERFUNC(BroadcastTo, BroadcastToInferShape)
{
    Tensor data;
    if (op.GetInputConstData("shape", data) != GRAPH_SUCCESS) {
        OP_LOGI(op.GetName().c_str(), "Get constValue failed of [shape]");
        Shape ashape = op.GetInputDesc("shape").GetShape();
        std::vector<int64_t> shapedims = ashape.GetDims();
        size_t dim_num = ashape.GetDimNum();

        DataType input_dtype = op.GetInputDesc("x").GetDataType();

        if (dim_num > 1) {
            OP_LOGE(op.GetName().c_str(), "The dim numbles of constnode are less than one.");
            return GRAPH_FAILED;
        }

        std::vector<int64_t> shape_vector;
        for (int64_t item = 0; item < shapedims[0]; ++item) {
            shape_vector.push_back(-1);
        }
        Shape input_shape(shape_vector);

        TensorDesc output_desc = op.GetOutputDesc("y");
        output_desc.SetShape(input_shape);
        output_desc.SetDataType(input_dtype);
        (void)op.UpdateOutputDesc("y", output_desc);

        return GRAPH_SUCCESS;
    }

    DataType data_type = data.GetTensorDesc().GetDataType();
    std::vector<int64_t> vec_dim;
    if (data_type == DT_INT32) {
        CaclDims<int32_t>(data, vec_dim);
    } else if (data_type == DT_INT64) {
        CaclDims<int64_t>(data, vec_dim);
    } else {
        return GRAPH_PARAM_INVALID;
    }
    OP_LOGI(op.GetName().c_str(), "the op infer shape and dtype");
    DataType input_dtype = op.GetInputDesc("x").GetDataType();

    TensorDesc td = op.GetOutputDesc("y");
    td.SetShape(Shape(vec_dim));
    td.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BroadcastTo, BroadcastToInferShape);
// --------------------BroadcastTo END-----------------------

// ------------------BroadcastToD------------------------
IMPLEMT_INFERFUNC(BroadcastToD, BroadcastToDInferShape)
{
    OP_LOGI(op.GetName().c_str(), "the op infer shape and dtype");
    DataType input_dtype = op.GetInputDesc("x").GetDataType();
    std::vector<int64_t> shape_out;
    if (ge::GRAPH_SUCCESS != op.GetAttr("shape", shape_out)) {
        OpsGetAttrErrReport(op.GetName(), "shape");
        OP_LOGE(op.GetName().c_str(), "GetOpAttr failed of BroadcastToD!");
        return GRAPH_FAILED;
    }
    if (shape_out.size() < DIM_SIZE1 || shape_out.size() > DIM_SIZE8) {
        OpsInputShapeDimErrReport(op.GetName(), "shape", Strcat(DIM_SIZE8), Strcat(DIM_SIZE1),
            Strcat(shape_out.size()));
        OP_LOGE(op.GetName().c_str(), "shape must be between 1 and 8.");
        return GRAPH_FAILED;
    }
    TensorDesc td = op.GetOutputDesc("y");
    td.SetShape(ge::Shape(shape_out));
    td.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", td);

    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(BroadcastToD, BroadcastToDInferShape);
// ----------------BroadcastToD END-------------------

// ---------------------DiagD-------------------------
COMMON_INFER_FUNC_REG(DiagD, ELMTWISE_INFER_SHAPEANDTYPE("assist", "y"));
// ---------------------DiagD_End---------------------

// ---------------------Diag--------------------------
IMPLEMT_COMMON_INFERFUNC(DiagInferShape)
{
    Shape shape = op.GetInputDesc("x").GetShape();
    DataType input_dtype = op.GetInputDesc("x").GetDataType();
    vector<int64_t> dimInfo = shape.GetDims();
    vector<int64_t> assitDimInfo;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < dimInfo.size(); ++j) {
            assitDimInfo.push_back(dimInfo[j]);
        }
    }

    shape = Shape(assitDimInfo);
    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    tensordesc_output.SetShape(shape);
    tensordesc_output.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", tensordesc_output);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Diag, DiagInferShape);
// ---------------------Diag END-------------------------------------

// ---------------------AscendPadding-------------------------------------
IMPLEMT_COMMON_INFERFUNC(AscendPaddingInferShape)
{
    Shape x_shape;
    auto x_desc = op.GetInputDesc(0);
    if (WithRankAtLeast(x_desc, 2, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input x rank must be at least 2, real rank is %lld.",
            x_desc.GetShape().GetDimNum());
        return GRAPH_FAILED;
    }

    auto x_rank = x_shape.GetDimNum();
    auto x_dims = x_shape.GetDims();

    if (x_dims[x_rank - 1] != 1) {
        OP_LOGE(op.GetName().c_str(), "the last dim of x must be 1, real dim is %lld.", x_dims[x_rank - 1]);
        return GRAPH_FAILED;
    }

    int32_t pad_dim_size;
    if (op.GetAttr("pad_dim_size", pad_dim_size) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get attr pad_dim_size error.");
        return GRAPH_FAILED;
    }
    if (pad_dim_size < 1) {
        OP_LOGE(op.GetName().c_str(), "pad_dim_size should be a positive value, real value is %d.", pad_dim_size);
        return GRAPH_PARAM_INVALID;
    }

    if (ReplaceDim(x_shape, x_rank - 1, pad_dim_size, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "failed to create y shape.");
        return GRAPH_FAILED;
    }
    auto y_desc = op.GetOutputDesc(0);
    y_desc.SetShape(x_shape);
    y_desc.SetDataType(x_desc.GetDataType());
    (void)op.UpdateOutputDesc("y", y_desc);

    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AscendPadding, AscendPaddingInferShape);
// ---------------------AscendPadding END-------------------------------------

// ---------------------EmbdingRankId-------------------------------------
IMPLEMT_COMMON_INFERFUNC(EmbeddingRankIdInferShape)
{
    Shape addr_shape;
    auto addr_desc = op.GetInputDesc(0);
    if (WithRank(addr_desc, 2, addr_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input addr_table rank must be at least 2, real rank is %lld.",
            addr_desc.GetShape().GetDimNum());
        return GRAPH_FAILED;
    }
    auto addr_rank = addr_shape.GetDimNum();
    auto addr_dims = addr_shape.GetDims();

    if (addr_dims[addr_rank - 1] != 3) {
        OP_LOGE(op.GetName().c_str(), "the last dim of addr_table must be 3, real dim is %lld.", addr_dims[addr_rank - 1]);
        return GRAPH_FAILED;
    }
    if (addr_dims[0] <= 0) {
        OP_LOGE(op.GetName().c_str(), "the first dim of addr_table must be >0, real dim is %lld.", addr_dims[0]);
        return GRAPH_FAILED;
    }
    Shape index_shape;
    auto index_desc = op.GetInputDesc(1);
    if (WithRank(index_desc, 1, index_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input index rank must be at least 1, real rank is %lld.",
            index_desc.GetShape().GetDimNum());
        return GRAPH_FAILED;
    }

    int32_t row_memory;
    if (op.GetAttr("row_memory", row_memory) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "get attr row_memory error.");
        return GRAPH_FAILED;
    }
    if (row_memory <= 0) {
        OP_LOGE(op.GetName().c_str(), "row_memory should be >0 , real value is %d.", row_memory);
        return GRAPH_PARAM_INVALID;
    }

    Shape out_shape;
    std::vector<int64_t> dims = index_shape.GetDims();
    if (ReplaceDim(addr_shape, 0, dims[0], out_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "failed to create rank_id shape.");
        return GRAPH_FAILED;
    }

    auto rankid_desc = op.GetOutputDesc(0);
    rankid_desc.SetShape(out_shape);
    rankid_desc.SetDataType(DT_UINT64);
    (void)op.UpdateOutputDesc("rank_id", rankid_desc);

    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(EmbeddingRankId, EmbeddingRankIdInferShape);
// ---------------------EmbeddingRankId END-------------------------------------
}
