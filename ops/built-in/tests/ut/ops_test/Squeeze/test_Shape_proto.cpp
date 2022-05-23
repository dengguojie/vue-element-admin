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
 * @file test_shape_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "util.h"
#include "graph/utils/op_desc_utils.h"
#include "array_ops.h"
#include "graph/ge_tensor.h"

class shape_ut : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "shape_ut SetUp" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "shape_ut TearDown" << std::endl;
  }
};

TEST_F(shape_ut, shape_success) {
  ge::op::Shape op("Shape");
  op.UpdateInputDesc("x", create_desc({1, 3, 2, 5}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {4};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(shape_ut, shape_unknow_dim) {
  ge::op::Shape op("Shape");
  op.UpdateInputDesc("x", create_desc({-1}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {1};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(shape_ut, shape_unknow_rank) {
  ge::op::Shape op("Shape");
  op.UpdateInputDesc("x", create_desc({-2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {-1};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}
