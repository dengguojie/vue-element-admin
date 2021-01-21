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

class SparseFillEmptyRowsTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseFillEmptyRowsTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseFillEmptyRowsTest TearDown" << std::endl;
  }
};

TEST_F(SparseFillEmptyRowsTest, InferShape) {
  ge::op::SparseFillEmptyRows op;
  op.UpdateInputDesc("indices", create_desc({5, 2}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({5}, ge::DT_INT64));
  op.UpdateInputDesc("dense_shape", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("default_value", create_desc({}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_indices_desc = op.GetOutputDesc("y_indices");
  EXPECT_EQ(y_indices_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_y_indices_shape = {-1, 2};
  EXPECT_EQ(y_indices_desc.GetShape().GetDims(), expected_y_indices_shape);

  auto y_values_desc = op.GetOutputDesc("y_values");
  EXPECT_EQ(y_values_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_y_values_shape = {-1};
  EXPECT_EQ(y_values_desc.GetShape().GetDims(), expected_y_values_shape);

  auto indicator_desc = op.GetOutputDesc("empty_row_indicator");
  EXPECT_EQ(indicator_desc.GetDataType(), ge::DT_BOOL);
  std::vector<int64_t> expected_indicator_shape = {-1};
  EXPECT_EQ(indicator_desc.GetShape().GetDims(), expected_indicator_shape);

  auto reverse_desc = op.GetOutputDesc("reverse_index_map");
  EXPECT_EQ(reverse_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_reverse_shape = {5};
  EXPECT_EQ(reverse_desc.GetShape().GetDims(), expected_reverse_shape);
}

