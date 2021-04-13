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
 * @file test_fifo_queue_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "set_ops.h"

class DenseToSparseSetOperationTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DenseToSparseSetOperationTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DenseToSparseSetOperationTest TearDown" << std::endl;
  }
};

TEST_F(DenseToSparseSetOperationTest, InferShape_01) {
  ge::op::DenseToSparseSetOperation op;
  op.UpdateInputDesc("x1", create_desc({3, 2}, ge::DT_INT64));
  op.UpdateInputDesc("x2_indices", create_desc({3, 2}, ge::DT_INT64));
  op.UpdateInputDesc("x2_values", create_desc({3}, ge::DT_INT64));
  op.UpdateInputDesc("x2_shape", create_desc({2}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto indices_desc = op.GetOutputDesc("y_indices");
  EXPECT_EQ(indices_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_indices_shape = {-1, 2};
  EXPECT_EQ(indices_desc.GetShape().GetDims(), expected_indices_shape);

  auto values_desc = op.GetOutputDesc("y_values");
  EXPECT_EQ(values_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_values_shape = {-1};
  EXPECT_EQ(values_desc.GetShape().GetDims(), expected_values_shape);

  auto shape_desc = op.GetOutputDesc("y_shape");
  EXPECT_EQ(shape_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_shape = {2};
  EXPECT_EQ(shape_desc.GetShape().GetDims(), expected_shape);
}

//error input num
TEST_F(DenseToSparseSetOperationTest, InferShape_02) {
  ge::op::DenseToSparseSetOperation op;
  op.UpdateInputDesc("x1", create_desc({3, 2}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//error x1 rank
TEST_F(DenseToSparseSetOperationTest, InferShape_03) {
  ge::op::DenseToSparseSetOperation op;
  op.UpdateInputDesc("x1", create_desc({3}, ge::DT_INT64));
  op.UpdateInputDesc("x2_indices", create_desc({3, 2}, ge::DT_INT64));
  op.UpdateInputDesc("x2_values", create_desc({3}, ge::DT_INT64));
  op.UpdateInputDesc("x2_shape", create_desc({2}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//error x2 shape dim[0]
TEST_F(DenseToSparseSetOperationTest, InferShape_04) {
  ge::op::DenseToSparseSetOperation op;
  op.UpdateInputDesc("x1", create_desc({3, 2}, ge::DT_INT64));
  op.UpdateInputDesc("x2_indices", create_desc({3, 2}, ge::DT_INT64));
  op.UpdateInputDesc("x2_values", create_desc({3}, ge::DT_INT64));
  op.UpdateInputDesc("x2_shape", create_desc({3}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DenseToSparseSetOperationTest, InferShape_05) {
  ge::op::DenseToSparseSetOperation op;
  op.UpdateInputDesc("x1", create_desc({-2}, ge::DT_INT64));
  op.UpdateInputDesc("x2_indices", create_desc({3, 2}, ge::DT_INT64));
  op.UpdateInputDesc("x2_values", create_desc({3}, ge::DT_INT64));
  op.UpdateInputDesc("x2_shape", create_desc({2}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}