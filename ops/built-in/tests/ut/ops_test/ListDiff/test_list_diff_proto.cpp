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
#include "array_ops.h"

class ListDiffTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ListDiffTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ListDiffTest TearDown" << std::endl;
  }
};

TEST_F(ListDiffTest, InferShape) {
  ge::op::ListDiff op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("y", create_desc({2}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_desc = op.GetOutputDesc("out");
  EXPECT_EQ(out_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_out_shape = {-1};
  EXPECT_EQ(out_desc.GetShape().GetDims(), expected_out_shape);

  auto idx_desc = op.GetOutputDesc("idx");
  EXPECT_EQ(idx_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_idx_shape = {-1};
  EXPECT_EQ(idx_desc.GetShape().GetDims(), expected_idx_shape);
}

