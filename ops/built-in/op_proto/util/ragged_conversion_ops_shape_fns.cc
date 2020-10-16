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
 * @file ragged_conversion_ops_shape_fns.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "ragged_conversion_ops_shape_fns.h"
#include <unordered_map>
#include "op_log.h"
#include "common_shape_fns.h"
namespace ge {
namespace {
int64_t MultiplyWithoutOverflow(const int64_t x, const int64_t y)
{
    const uint64_t ux = x;
    const uint64_t uy = y;
    const uint64_t uxy = ux * uy;

    if ((ux | uy) >> 32 != 0) {
        if (ux != 0 && uxy / ux != uy)
            return -1;
    }

    return static_cast<int64_t>(uxy);
}

graphStatus GetRowPartitionTypes(Operator &op, std::vector<RowPartitionType> &row_partition_types)
{
    std::vector<std::string> partition_types;
    if (op.GetAttr("row_partition_types", partition_types) != GRAPH_SUCCESS) {
         OP_LOGE(op.GetName().c_str(), "Op get attr row_partition_types failed");
        return GRAPH_FAILED;
    }

    const auto string_to_type =
        new std::unordered_map<std::string, RowPartitionType>({ { "FIRST_DIM_SIZE", RowPartitionType::FIRST_DIM_SIZE },
        { "VALUE_ROWIDS", RowPartitionType::VALUE_ROWIDS },
        { "ROW_LENGTHS", RowPartitionType::ROW_LENGTHS },
        { "ROW_SPLITS", RowPartitionType::ROW_SPLITS },
        { "ROW_LIMITS", RowPartitionType::ROW_LIMITS },
        { "ROW_STARTS", RowPartitionType::ROW_STARTS } });

    for (const std::string &type_str : partition_types) {
        const auto iter = string_to_type->find(type_str);
        if (iter == string_to_type->end()) {
            OP_LOGE(op.GetName().c_str(), "Unknown string for partition info type.");
            return GRAPH_FAILED;
        }
        row_partition_types.push_back(iter->second);
    }
    return GRAPH_SUCCESS;
}

int32_t GetRaggedRank(const std::vector<RowPartitionType> &partition_types)
{
    if (partition_types.empty()) {
        return 0;
    }
    if (partition_types[0] == RowPartitionType::FIRST_DIM_SIZE) {
        return partition_types.size() - 1;
    }
    return partition_types.size();
}

graphStatus ValidateDefaultValueShape(const TensorShape &default_value_shape,
                                      const TensorShape &value_shape,
                                      const char* op_name)
{
    if (default_value_shape.unknown_rank || value_shape.unknown_rank) {
        return GRAPH_SUCCESS;
    }

    if (default_value_shape.dims.size() > value_shape.dims.size()) {
         OP_LOGE(op_name, "default_value_shape must have no more dimensions than the value.");
        return GRAPH_FAILED;
    }

    for (size_t i = 0; i < std::min(default_value_shape.dims.size(), value_shape.dims.size() - 1); ++i) {
        if (default_value_shape.dims[i].size >= 0 && value_shape.dims[i + 1].size >= 0 &&
            default_value_shape.dims[i].size != 1 && default_value_shape.dims[i].size != value_shape.dims[i + 1].size) {
             OP_LOGE(op_name, "default_value_shape and value_shape do not match on dimension.");
            return GRAPH_FAILED;
        }
    }

    return GRAPH_SUCCESS;
}
} //  namespace

graphStatus MakeShapeFromShapeTensorTreatScalarAsUnknownShape(const Tensor &tensor,
                                                              Shape &out, const char* op_name)
{
    TensorDesc shape_data_desc = tensor.GetTensorDesc();
    Shape shape_data_shape = shape_data_desc.GetShape();
    std::vector<int64_t> dims = shape_data_shape.GetDims();
    DataType data_type = shape_data_desc.GetDataType();

    size_t rank_size = 1;
    if(!((dims.size() <= rank_size) || (dims == ge::UNKNOWN_SHAPE))) {
        OP_LOGE(op_name, "Shape's rank must be at most %lld, but it is %u", rank_size, dims.size());
        return GRAPH_FAILED;
    }

    if (dims.size() == 0) {
        if (data_type == DT_INT32) {
            const int32_t *shape_data = reinterpret_cast<const int32_t *>(tensor.GetData());
            if (shape_data[0] != -1) {
                 OP_LOGE(op_name, "if its rank 0 it must have value -1");
                return GRAPH_FAILED;
            }
        } else if (data_type == DT_INT64) {
            const int64_t *shape_data = reinterpret_cast<const int64_t *>(tensor.GetData());
            if (shape_data[0] != -1) {
                 OP_LOGE(op_name, "if its rank 0 it must have value -1");
                return GRAPH_FAILED;
            }
        } else {
             OP_LOGE(op_name, "Data type invalid, should be DT_INT32 or DT_INT64");
            return GRAPH_FAILED;
        }
        out = Shape(ge::UNKNOWN_SHAPE);
        return GRAPH_SUCCESS;
    }

    if (MakeShapeFromShapeTensor(tensor, out, op_name) != GRAPH_SUCCESS) {
        OP_LOGE(op_name, "MakeShapeFromShapeTensor failed");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

void ShapeHandleToTensorShape(Shape handle, TensorShape &shape_info)
{
    if (!RankKnown(handle)) {
        shape_info.unknown_rank = true;
        return;
    }

    for (size_t i = 0; i < handle.GetDimNum(); ++i) {
        int64_t dim = handle.GetDim(i);
        Dim temp_dim;
        if (ValueKnown(handle, i)) {
            temp_dim.size = dim;
        } else {
            temp_dim.size = -1;
        }
        shape_info.dims.emplace_back(temp_dim);
    }
}

graphStatus CombineRaggedTensorToTensorShapes(int32_t ragged_rank, const TensorShape &shape,
    const TensorShape &value_shape, TensorShape &output_shape, const char* op_name)
{
    if (value_shape.unknown_rank && shape.unknown_rank) {
        output_shape.dims.clear();
        output_shape.unknown_rank = true;
        return GRAPH_SUCCESS;
    }

    if (shape.unknown_rank) {
        while (output_shape.dims.size() < ragged_rank + value_shape.dims.size()) {
            Dim temp_dim;
            temp_dim.size = -1;
            output_shape.dims.emplace_back(temp_dim);
        }
    } else {
        output_shape = shape;
    }
    if (value_shape.unknown_rank) {
        return GRAPH_SUCCESS;
    }

    if (ragged_rank + value_shape.dims.size() != output_shape.dims.size()) {
         OP_LOGE(op_name, "Value shape and ragged_rank dont have a consistent number of dimensions.");
        return GRAPH_FAILED;
    }

    for (size_t i = 1; i < value_shape.dims.size(); ++i) {
        const Dim value_dim = value_shape.dims[i];
        Dim output_shape_dim = output_shape.dims.at(output_shape.dims.size() - value_shape.dims.size() + i);

        if (value_dim.size >= 0) {
            if (output_shape_dim.size >= 0) {
                if (output_shape_dim.size != value_dim.size) {
                    OP_LOGE(op_name, "Value and shape dimension are inconsistent.");
                    return GRAPH_FAILED;
                }
            } else {
                output_shape_dim.size = value_dim.size;
            }
        }
    }

    return GRAPH_SUCCESS;
}


graphStatus IsValidShape(const TensorShape &shape, const char* op_name)
{
    int64_t num_elements = 1;
    size_t max_dimensions = 254;
    if (shape.dims.size() > max_dimensions) {
         OP_LOGE(op_name, "Shape has too many dimensions.");
        return GRAPH_FAILED;
    }
    for (const auto &d : shape.dims) {
        if (d.size == -1) {
            num_elements = -1;
        } else {
            num_elements = MultiplyWithoutOverflow(num_elements, d.size);
            if (num_elements < 0) {
                 OP_LOGE(op_name, "Shape is too large (more than 2**63 - 1 entries).");
                return GRAPH_FAILED;
            }
        }
    }
    return GRAPH_SUCCESS;
}

graphStatus MakeShapeFromTensorShape(const TensorShape &input_shape, Shape &out, const char * op_name)
{
    if (IsValidShape(input_shape, op_name) != GRAPH_SUCCESS) {
         OP_LOGE(op_name, "check input shape is valid shape failed.");
        return GRAPH_FAILED;
    }
    if (input_shape.unknown_rank) {
        out = Shape(ge::UNKNOWN_SHAPE);
        return GRAPH_SUCCESS;
    }
    std::vector<int64_t> dims;
    for (const auto &d : input_shape.dims) {
        dims.emplace_back(d.size);
    }
    out = Shape(dims);
    return GRAPH_SUCCESS;
}

graphStatus RaggedTensorToTensorShapeFn(Operator &op)
{
    TensorShape shape;
    {
        Shape shape_handle;
        Tensor tensor;
        if (op.GetInputConstData("shape", tensor) != GRAPH_SUCCESS) {
            OP_LOGE(op.GetName().c_str(), "input orig_input_tensor_shape GetInputConstData failed");
            return GRAPH_FAILED;
        }
        if (MakeShapeFromShapeTensorTreatScalarAsUnknownShape(tensor, shape_handle, op.GetName().c_str()) != GRAPH_SUCCESS) {
            OP_LOGE(op.GetName().c_str(), "makeShapeFromShapeTensorTreatScalarAsUnknownShape failed");
            return GRAPH_FAILED;
        }

        ShapeHandleToTensorShape(shape_handle, shape);
    }

    std::vector<RowPartitionType> row_partition_types;
    if (GetRowPartitionTypes(op, row_partition_types) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "GetRowPartitionTypes failed");
        return GRAPH_FAILED;
    }

    int32_t ragged_rank = GetRaggedRank(row_partition_types);

    TensorShape value_shape;
    ShapeHandleToTensorShape(op.GetInputDesc("values").GetShape(), value_shape);

    TensorShape default_value_shape;
    ShapeHandleToTensorShape(op.GetInputDesc("default_value").GetShape(), default_value_shape);

    if (ValidateDefaultValueShape(default_value_shape, value_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Validate default value shape failed");
        return GRAPH_FAILED;
    }

    TensorShape output_shape;
    if (CombineRaggedTensorToTensorShapes(ragged_rank, shape, value_shape, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "CombineRaggedTensorToTensorShapes failed");
        return GRAPH_FAILED;
    }

    Shape output_shape_handle;
    if (MakeShapeFromTensorShape(output_shape, output_shape_handle, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "MakeShapeFromShapeProto failed");
        return GRAPH_FAILED;
    }

    TensorDesc out_desc = op.GetOutputDesc("result");
    out_desc.SetShape(output_shape_handle);
    out_desc.SetDataType(op.GetInputDesc("values").GetDataType());
    if (op.UpdateOutputDesc("result", out_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update result desc failed.");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}
} // namespace ge
