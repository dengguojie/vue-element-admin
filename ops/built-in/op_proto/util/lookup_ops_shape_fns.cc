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
 * @file lookup_ops_shape_fns.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "lookup_ops_shape_fns.h"
#include <vector>
#include <limits>
#include "op_log.h"
#include "common_shape_fns.h"
using namespace std;

namespace ge {
graphStatus ValidateTableResourceHandle(Shape keys, std::vector<ShapeAndType> handleData,
    ShapeAndType &output_shape_and_type, bool is_lookup, const char* op_name)
{
    Shape unknown_shape(ge::UNKNOWN_SHAPE);
    if (handleData.size() != 2) {
        output_shape_and_type.SetShape(unknown_shape);
        output_shape_and_type.SetType(DT_UNDEFINED);
    } else {
        const ShapeAndType &key_shape_and_type = handleData[0];
        const ShapeAndType &value_shape_and_type = handleData[1];
        // here need to check key_dtype and value_dtype
        // but can not get the attr type for key and value
        output_shape_and_type.SetType(value_shape_and_type.GetDataType());
        if (is_lookup) {
            if ((RankKnown(key_shape_and_type.GetShape()) == GRAPH_SUCCESS) && (RankKnown(keys) == GRAPH_SUCCESS)) {
                int keys_rank = keys.GetDims().size();
                int keys_suffix_rank = key_shape_and_type.GetShape().GetDims().size();
                if (keys_rank < keys_suffix_rank) {
                     OP_LOGE(op_name, "Expected keys to have suffix");
                    return GRAPH_FAILED;
                }
                for (int d = 0; d < keys_suffix_rank; ++d) {
                    int new_dim = key_shape_and_type.GetShape().GetDim(d);
                    if (ReplaceDim(keys, keys_rank - keys_suffix_rank + d, new_dim, keys, op_name) == GRAPH_FAILED) {
                        return GRAPH_FAILED;
                    }
                }
                vector<int64_t> keys_prefix_vec;
                keys_prefix_vec.reserve(keys_rank - keys_suffix_rank);
                for (int d = 0; d < keys_rank - keys_suffix_rank; ++d) {
                    keys_prefix_vec.push_back(keys.GetDim(d));
                }
                Shape keys_prefix = Shape(keys_prefix_vec);
                Shape temp_shape = output_shape_and_type.GetShape();
                if (Concatenate(keys_prefix, value_shape_and_type.GetShape(), temp_shape) == GRAPH_FAILED) {
                    return GRAPH_FAILED;
                }
                output_shape_and_type.SetShape(temp_shape);
            } else {
                output_shape_and_type.SetShape(unknown_shape);
            }
        } else {
            Shape temp_shape = output_shape_and_type.GetShape();
            if (Concatenate(keys, value_shape_and_type.GetShape(), temp_shape) == GRAPH_FAILED) {
                return GRAPH_FAILED;
            }
            output_shape_and_type.SetShape(temp_shape);
        }
    }
    return GRAPH_SUCCESS;
}
} // namespace ge
