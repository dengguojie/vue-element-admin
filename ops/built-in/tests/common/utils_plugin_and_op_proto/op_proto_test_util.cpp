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
#include <iostream>
#include "op_proto_test_util.h"

ge::TensorDesc create_desc(std::initializer_list<int64_t> shape_dims,
                           ge::DataType dt) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);
  return tensorDesc;
}

ge::TensorDesc create_desc_with_ori(std::initializer_list<int64_t> shape_dims,
                           ge::DataType dt,
                           ge::Format format,
                           std::initializer_list<int64_t> ori_shape_dims,
                           ge::Format ori_format) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginFormat(ori_format);
  return tensorDesc;
}

ge::TensorDesc create_desc_with_original_shape(std::initializer_list<int64_t> shape_dims,
                                               ge::DataType dt,
                                               ge::Format format,
                                               std::initializer_list<int64_t> ori_shape_dims,
                                               ge::Format ori_format) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(ori_shape);
  tensorDesc.SetOriginFormat(ori_format);
  return tensorDesc;
}

ge::TensorDesc create_desc_shape_range(
    std::initializer_list<int64_t> shape_dims,
    ge::DataType dt,
    ge::Format format,
    std::initializer_list<int64_t> ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginFormat(ori_format);
  tensorDesc.SetShapeRange(shape_range);
  return tensorDesc;
}

ge::TensorDesc create_desc_shape_range(
    const std::vector<int64_t>& shape_dims,
    ge::DataType dt,
    ge::Format format,
    const std::vector<int64_t>& ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginFormat(ori_format);
  tensorDesc.SetShapeRange(shape_range);
  return tensorDesc;
}

ge::TensorDesc create_desc_shape_and_origin_shape_range(
    const std::vector<int64_t>& shape_dims,
    ge::DataType dt,
    ge::Format format,
    const std::vector<int64_t>& ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(ori_shape);
  tensorDesc.SetOriginFormat(ori_format);
  tensorDesc.SetShapeRange(shape_range);
  return tensorDesc;
}

ge::TensorDesc create_desc_shape_and_origin_shape_range(
    std::initializer_list<int64_t> shape_dims,
    ge::DataType dt,
    ge::Format format,
    std::initializer_list<int64_t> ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(ori_shape);
  tensorDesc.SetOriginFormat(ori_format);
  tensorDesc.SetShapeRange(shape_range);
  return tensorDesc;
}