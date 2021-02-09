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

class CheckNumericsTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CheckNumericsTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CheckNumericsTest TearDown" << std::endl;
  }
};

TEST_F(CheckNumericsTest, InferShape) {
  ge::op::CheckNumerics op;
  op.UpdateInputDesc("tensor", create_desc({5}, ge::DT_FLOAT));
  op.SetAttr("message", "1");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("output");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

