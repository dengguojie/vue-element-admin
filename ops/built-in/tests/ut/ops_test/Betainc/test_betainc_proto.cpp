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

class BetaincTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BetaincTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BetaincTest TearDown" << std::endl;
  }
};

TEST_F(BetaincTest, InferShape_01) {
  ge::op::Betainc op;
  op.UpdateInputDesc("a", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("b", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("x", create_desc({1}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto z_desc = op.GetOutputDesc("z");
  EXPECT_EQ(z_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_z_shape = {1};
  EXPECT_EQ(z_desc.GetShape().GetDims(), expected_z_shape);
}

//error size rank
TEST_F(BetaincTest, InferShape_02) {
  ge::op::Bincount op;
  op.UpdateInputDesc("array", create_desc({2, 2}, ge::DT_INT32));
  op.UpdateInputDesc("size", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("weights", create_desc({}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}