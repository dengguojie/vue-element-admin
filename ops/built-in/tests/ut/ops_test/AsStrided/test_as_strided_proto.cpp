/**
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_as_strided_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "transformation_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"

namespace as_strided_test_proto {
class AsStrided: public testing::Test {
 protected:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
  }
};

template <typename T>
void test(vector<int64_t> input_shape,
          vector<int64_t> ori_shape,
          vector<T> size,
          vector<T> stride,
          T storage_offset,
          vector<std::pair<int64_t, int64_t>> shape_range,
          vector<int64_t> &output_shape,
          vector<std::pair<int64_t, int64_t>> &output_range) {
  ge::op::AsStrided op;
  auto tensor_desc = create_desc_shape_range(input_shape, ge::DT_FLOAT16, ge::FORMAT_ND, ori_shape, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);

  auto dtype = ge::DT_INT32;
  if (sizeof(T) == sizeof(int64_t)) {
    dtype = ge::DT_INT64;
  }

  if (!size.empty()) {
    ge::Tensor constTensorSize;
    ge::TensorDesc constDescSize(ge::Shape(), ge::FORMAT_ND, dtype);
    constDescSize.SetSize(size.size() * sizeof(T));
    constTensorSize.SetTensorDesc(constDescSize);
    constTensorSize.SetData((uint8_t*)size.data(), size.size() * sizeof(T));
    auto input_size = ge::op::Constant().set_attr_value(constTensorSize);
    op.set_input_size(input_size);
    auto descSize = op.GetInputDesc("size");
    descSize.SetDataType(dtype);
    op.UpdateInputDesc("size", descSize);
  }

  if (!stride.empty()) {
    ge::Tensor constTensorStride;
    ge::TensorDesc constDescStride(ge::Shape(), ge::FORMAT_ND, dtype);
    constDescStride.SetSize(stride.size() * sizeof(T));
    constTensorStride.SetTensorDesc(constDescStride);
    constTensorStride.SetData((uint8_t*)stride.data(), stride.size() * sizeof(T));
    auto input_stride = ge::op::Constant().set_attr_value(constTensorStride);
    op.set_input_stride(input_stride);
    auto descStride = op.GetInputDesc("stride");
    descStride.SetDataType(dtype);
    op.UpdateInputDesc("stride", descStride);
  }

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  output_shape = output_desc.GetShape().GetDims();
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
}

TEST_F(AsStrided, case1) {
  vector<int64_t> output_shape;
  vector<int64_t> expect_output_shape;
  expect_output_shape.push_back(2000);
  expect_output_shape.push_back(100);
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int32_t>({-1, 1000}, {-1, 1000}, {2000, 100}, {1000, 10}, 0, {{2000, 2000}, {1000, 1000}}, output_shape, output_range);
  EXPECT_EQ(expect_output_shape, output_shape);
}

TEST_F(AsStrided, case2) {
  vector<int64_t> output_shape;
  vector<int64_t> expect_output_shape;
  expect_output_shape.push_back(-2);
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int32_t>({-1, 1000}, {-1, 1000}, {}, {}, 0, {{2000, 2000}, {1000, 1000}}, output_shape, output_range);
  EXPECT_EQ(expect_output_shape, output_shape);
}
};
