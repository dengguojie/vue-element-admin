/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef OP_IR_TEST__OP_TEST_UTIL_H_
#define OP_IR_TEST__OP_TEST_UTIL_H_
#include <iostream>
#include "graph/types.h"
#include "graph/tensor.h"


ge::TensorDesc create_desc(std::initializer_list<int64_t> shape_dims,
                           ge::DataType dt=ge::DT_FLOAT);

ge::TensorDesc create_desc_with_ori(std::initializer_list<int64_t> shape_dims,
                                    ge::DataType dt=ge::DT_FLOAT,
                                    ge::Format format=ge::FORMAT_ND,
                                    std::initializer_list<int64_t> ori_shape_dims={},
                                    ge::Format ori_format=ge::FORMAT_ND);
ge::TensorDesc create_desc_shape_range(
    std::initializer_list<int64_t> shape_dims,
    ge::DataType dt,
    ge::Format format,
    std::initializer_list<int64_t> ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range);

ge::TensorDesc create_desc_shape_range(
    const std::vector<int64_t>& shape_dims,
    ge::DataType dt,
    ge::Format format,
    const std::vector<int64_t>& ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range);
ge::TensorDesc create_desc_with_original_shape(std::initializer_list<int64_t> shape_dims,
                                               ge::DataType dt,
                                               ge::Format format,
                                               std::initializer_list<int64_t> ori_shape_dims,
                                               ge::Format ori_format);
ge::TensorDesc create_desc_shape_and_origin_shape_range(
    std::initializer_list<int64_t> shape_dims,
    ge::DataType dt,
    ge::Format format,
    std::initializer_list<int64_t> ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range);
    
ge::TensorDesc create_desc_shape_and_origin_shape_range(
    const std::vector<int64_t>& shape_dims,
    ge::DataType dt,
    ge::Format format,
    const std::vector<int64_t>& ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range);
#endif //OP_IR_TEST__OP_TEST_UTIL_H_
