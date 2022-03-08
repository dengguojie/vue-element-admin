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
 * @file test_dynamic_approximate_equal_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class clip_by_value : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "clip_by_value SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "clip_by_value TearDown" << std::endl;
  }
};

TEST_F(clip_by_value, clip_by_value_infershape_diff_test_1) {
  ge::op::ClipByValue op;
  op.UpdateInputDesc("x", create_desc_shape_range({-1, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {30, 1},
                                                           ge::FORMAT_NHWC, {{1,50}, {1,1}}));
  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(2);
  op.UpdateInputDesc("x", tensor_x);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y1_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT);//output data type

  std::vector<int64_t> expected_output_shape = {-1,1};
  EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape); //output shape

  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);

  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1,50}, {1,1}};
  EXPECT_EQ(output_shape_range, expected_shape_range);//output shape range
}
TEST_F(clip_by_value, clip_by_value_infershape_diff_test_2) {
  ge::op::ClipByValue op;
  op.UpdateInputDesc("x", create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1,1}}));
  op.UpdateInputDesc("clip_value_min", create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1,3}}));
  op.UpdateInputDesc("clip_value_max", create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1,3}}));
  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  ge::TensorDesc tensor_clip_value_min = op.GetInputDesc("clip_value_min");
  ge::TensorDesc tensor_clip_value_max = op.GetInputDesc("clip_value_max");
  op.UpdateInputDesc("x", tensor_x);
  op.UpdateInputDesc("clip_value_min", tensor_clip_value_min);
  op.UpdateInputDesc("clip_value_max", tensor_clip_value_max);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y1_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT);//output data type

  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape); //output shape

  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);

  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1,3}};
  EXPECT_EQ(output_shape_range, expected_shape_range);//output shape range
}

TEST_F(clip_by_value, clip_by_value_infershape_diff_test_3) {
  ge::op::ClipByValue op;
  op.UpdateInputDesc("x",
                     create_desc_shape_range({1}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 1}}));
  op.UpdateInputDesc("clip_value_min",
                     create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));
  op.UpdateInputDesc("clip_value_max",
                     create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));
  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  ge::TensorDesc tensor_clip_value_min = op.GetInputDesc("clip_value_min");
  ge::TensorDesc tensor_clip_value_max = op.GetInputDesc("clip_value_max");
  op.UpdateInputDesc("x", tensor_x);
  op.UpdateInputDesc("clip_value_min", tensor_clip_value_min);
  op.UpdateInputDesc("clip_value_max", tensor_clip_value_max);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(clip_by_value, clip_by_value_infershape_diff_test_4) {
  ge::op::ClipByValue op;
  op.UpdateInputDesc("x",
                     create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 1}}));
  op.UpdateInputDesc("clip_value_min",
                     create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));
  op.UpdateInputDesc("clip_value_max",
                     create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(clip_by_value, clip_by_value_infershape_diff_test_5) {
  ge::op::ClipByValue op;
  op.UpdateInputDesc("x",
                     create_desc_shape_range({3, 1, 5}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 1}}));
  op.UpdateInputDesc("clip_value_min",
                     create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));
  op.UpdateInputDesc("clip_value_max",
                     create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {3, 1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(clip_by_value, clip_by_value_infershape_diff_test_6) {
  ge::op::ClipByValue op;
  op.UpdateInputDesc("x",
                     create_desc_shape_range({3, 1, 5}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 1}}));
  op.UpdateInputDesc("clip_value_min",
                     create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));
  op.UpdateInputDesc("clip_value_max",
                     create_desc_shape_range({5}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {3, 1, 5};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(clip_by_value, clip_by_value_infershape_diff_test_7) {
  ge::op::ClipByValue op;
  op.UpdateInputDesc("x",
                     create_desc_shape_range({3, 1, 5}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 1}}));
  op.UpdateInputDesc("clip_value_min",
                     create_desc_shape_range({1, 5}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));
  op.UpdateInputDesc("clip_value_max",
                     create_desc_shape_range({5}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {3, 1, 5};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(clip_by_value, clip_by_value_infershape_diff_test_8) {
  ge::op::ClipByValue op;
  op.UpdateInputDesc("x",
                     create_desc_shape_range({3, 1, 5}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 1}}));
  op.UpdateInputDesc("clip_value_min",
                     create_desc_shape_range({1, 6}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));
  op.UpdateInputDesc("clip_value_max",
                     create_desc_shape_range({5}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(clip_by_value, clip_by_value_infershape_diff_test_9) {
  ge::op::ClipByValue op;
  op.UpdateInputDesc("x",
                     create_desc_shape_range({3, 1, 5}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 1}}));
  op.UpdateInputDesc("clip_value_min",
                     create_desc_shape_range({1, -1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));
  op.UpdateInputDesc("clip_value_max",
                     create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(clip_by_value, clip_by_value_infershape_diff_test_10) {
  ge::op::ClipByValue op;
  op.UpdateInputDesc("x",
                     create_desc_shape_range({3, 1, 5}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 1}}));
  op.UpdateInputDesc("clip_value_min",
                     create_desc_shape_range({0, -1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));
  op.UpdateInputDesc("clip_value_max",
                     create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(clip_by_value, clip_by_value_infershape_diff_test_11) {
  ge::op::ClipByValue op;
  op.UpdateInputDesc("x",
                     create_desc_shape_range({3, 1, 5}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 1}}));
  op.UpdateInputDesc("clip_value_min",
                     create_desc_shape_range({0, -1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));
  op.UpdateInputDesc("clip_value_max",
                     create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC, {{1, 3}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {3, 0, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
