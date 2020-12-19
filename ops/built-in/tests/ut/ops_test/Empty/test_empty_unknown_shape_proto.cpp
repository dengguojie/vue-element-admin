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
 * @file test_empty_unknown_shape.cpp
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

class EMPTY_UNKNOWN_SHAPE_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "EMPTY_UNKNOWN_SHAPE_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "EMPTY_UNKNOWN_SHAPE_UT TearDown" << std::endl;
  }
};

TEST_F(EMPTY_UNKNOWN_SHAPE_UT, InferShape) {
  ge::op::Empty op;
  op.UpdateInputDesc("x", create_desc({-2}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_INT64);
  op.SetAttr("init", true);
  op.InferShapeAndType();

  auto y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_y_shape = {};
  EXPECT_EQ(y_desc.GetShape().GetDims(), expected_y_shape);
}
