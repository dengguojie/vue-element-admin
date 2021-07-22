/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_Shrink_proto.cpp
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

const static char* input_x = "input_x";
class ShrinkTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ShrinkTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ShrinkTest TearDown" << std::endl;
  }
};

TEST_F(ShrinkTest, ShrinkTest_1) {
  ge::op::Shrink op;

  op.UpdateInputDesc(input_x,
    create_desc_shape_range(
     {-1, -1},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {1, 1},
     ge::FORMAT_ND, {{1, 10}, {1, 10}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ShrinkTest, ShrinkTest_2) {
  ge::op::Shrink op;

  op.UpdateInputDesc(input_x,
    create_desc_shape_range(
     {-1, -1},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {1, 1},
     ge::FORMAT_ND, {{1, 10}, {1, 10}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ShrinkTest, ShrinkTest_3) {
  ge::op::Shrink op;

  op.UpdateInputDesc(input_x,
    create_desc_shape_range(
     {-1, -1, -1},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {1, 1, 1},
     ge::FORMAT_ND, {{1, 10}, {1, 10}, {1, 10}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ShrinkTest, ShrinkTest_4) {
  ge::op::Shrink op;

  op.UpdateInputDesc(input_x,
    create_desc_shape_range(
     {-1, -1, -1},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {1, 1, 1},
     ge::FORMAT_ND, {{1, 10}, {1, 10}, {1, 10}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}