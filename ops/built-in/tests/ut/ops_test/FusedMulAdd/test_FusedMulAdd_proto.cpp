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
 * @file test_truncate_div_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class FusedMulAdd : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FusedMulAdd SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FusedMulAdd TearDown" << std::endl;
  }
};

TEST_F(FusedMulAdd, FusedMulAdd_infershape_test_1) {
  ge::op::FusedMulAdd op;
  op.UpdateInputDesc("x1", create_desc_shape_range({-1, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {30, 1},
                                                           ge::FORMAT_NHWC, {{1,50}, {1,1}}));
  ge::TensorDesc tensor_x = op.GetInputDesc("x1");
  tensor_x.SetRealDimCnt(2);
  op.UpdateInputDesc("x1", tensor_x);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y1_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_output_shape = {-1,1};
  EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);

  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1,50}, {1,1}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(FusedMulAdd, FusedMulAdd_infershape_test_2) {
  ge::op::FusedMulAdd op;

  op.UpdateInputDesc("x1", create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1,1}}));
  op.UpdateInputDesc("x2", create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1,3}}));
  op.UpdateInputDesc("x3", create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1,3}}));
  ge::TensorDesc tensor_x = op.GetInputDesc("x1");
  ge::TensorDesc tensor_clip_value_min = op.GetInputDesc("x2");
  ge::TensorDesc tensor_clip_value_max = op.GetInputDesc("x3");
  op.UpdateInputDesc("x1", tensor_x);
  op.UpdateInputDesc("x2", tensor_clip_value_min);
  op.UpdateInputDesc("x3", tensor_clip_value_max);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y1_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);

  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1,3}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(FusedMulAdd, FusedMulAdd_infershape_test_3) {
  ge::op::FusedMulAdd op;

  op.UpdateInputDesc("x1",
                     create_desc_shape_range({1}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 1}}));
  op.UpdateInputDesc("x2",
                     create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));
  op.UpdateInputDesc("x3",
                     create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, false);
}

TEST_F(FusedMulAdd, FusedMulAdd_infershape_test_4) {
  ge::op::FusedMulAdd op;

  op.UpdateInputDesc("x1", 
                     create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 1}}));
  op.UpdateInputDesc("x2",
                     create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));
  op.UpdateInputDesc("x3",
                     create_desc_shape_range({-1}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, false);
}