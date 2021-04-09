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
#include "sparse_ops.h"

class SparseAddTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseAddTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseAddTest TearDown" << std::endl;
  }
};

TEST_F(SparseAddTest, InferShape_01) {
  ge::op::SparseAdd op;
  op.UpdateInputDesc("x1_indices", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("x1_values", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("x1_shape", create_desc({3}, ge::DT_INT64));
  op.UpdateInputDesc("x2_indices", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("x2_values", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("x2_shape", create_desc({3}, ge::DT_INT64));
  op.UpdateInputDesc("thresh", create_desc({}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto sum_indices_desc = op.GetOutputDesc("sum_indices");
  EXPECT_EQ(sum_indices_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_sum_indices_shape = {-1, 3};
  EXPECT_EQ(sum_indices_desc.GetShape().GetDims(), expected_sum_indices_shape);

  auto sum_values_desc = op.GetOutputDesc("sum_values");
  EXPECT_EQ(sum_values_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_sum_values_shape = {-1};
  EXPECT_EQ(sum_values_desc.GetShape().GetDims(), expected_sum_values_shape);

  auto sum_shape_desc = op.GetOutputDesc("sum_shape");
  EXPECT_EQ(sum_shape_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_sum_shape_shape = {3};
  EXPECT_EQ(sum_shape_desc.GetShape().GetDims(), expected_sum_shape_shape);
}

//error x1_shape rank
TEST_F(SparseAddTest, InferShape_02) {
  ge::op::SparseAdd op;
  op.UpdateInputDesc("x1_indices", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("x1_values", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("x1_shape", create_desc({3, 1}, ge::DT_INT64));
  op.UpdateInputDesc("x2_indices", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("x2_values", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("x2_shape", create_desc({3}, ge::DT_INT64));
  op.UpdateInputDesc("thresh", create_desc({}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}