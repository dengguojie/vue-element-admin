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
#include "math_ops.h"

class SparseSegmentSumTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseSegmentSumTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseSegmentSumTest TearDown" << std::endl;
  }
};

TEST_F(SparseSegmentSumTest, InferShape) {
  ge::op::SparseSegmentSum op;
  op.UpdateInputDesc("x", create_desc({4, 2}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("segment_ids", create_desc({4}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_y_shape = {-1, 2};
  EXPECT_EQ(y_desc.GetShape().GetDims(), expected_y_shape);
}

