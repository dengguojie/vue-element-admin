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
#include "linalg_ops.h"

class CholeskyGradTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CholeskyGradTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CholeskyGradTest TearDown" << std::endl;
  }
};

TEST_F(CholeskyGradTest, InferShape) {
  ge::op::CholeskyGrad op;
  op.UpdateInputDesc("x", create_desc({1, 1, 2, 2}, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({1, 1, 2, 2}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {1, 1, 2, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

