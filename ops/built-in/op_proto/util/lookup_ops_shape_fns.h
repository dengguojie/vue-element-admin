/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file lookup_ops_shape_fns.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef LOOKUP_OPS_SHAPE_FNS_H
#define LOOKUP_OPS_SHAPE_FNS_H

#include <vector>
#include "graph/tensor.h"
#include "graph/inference_context.h"

namespace ge {
/**
 * Validate table resource handle
 * @param keys keys of the shape
 * @param handleData vector of handle data
 * @param output_shape_and_type shape and type that created
 * @param is_lookup if is lookup
 * @return status whether this operation success
 */
graphStatus ValidateTableResourceHandle(
    Shape keys,
    std::vector<ShapeAndType> handleData,
    ShapeAndType &output_shape_and_type,
    bool is_lookup, const char* op_name);
}   // namespace ge

#endif  // LOOKUP_OPS_SHAPE_FNS_H
