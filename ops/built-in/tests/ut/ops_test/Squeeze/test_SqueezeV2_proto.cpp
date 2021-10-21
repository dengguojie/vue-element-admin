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
 * @file test_Unsqueezev2_proto.cpp
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

class SqueezeV2_ut : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SqueezeV2_ut SetUp" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "SqueezeV2_ut TearDown" << std::endl;
  }
};


TEST_F(SqueezeV2_ut, squeezev2_negative_axis) {
  ge::op::SqueezeV2 op("SqueezeV2");
  op.UpdateInputDesc("x", create_desc({1, 3, 2, 5}, ge::DT_INT32));
  op.SetAttr("axis", {-4});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {3, 2, 5};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(SqueezeV2_ut, squeezev2_one_axis) {
  ge::op::SqueezeV2 op("SqueezeV2");
  op.UpdateInputDesc("x", create_desc({3, 1, 4, 5}, ge::DT_INT32));
  op.SetAttr("axis", {1});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {3, 4, 5};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(SqueezeV2_ut, squeezev2_two_axis) {
  ge::op::SqueezeV2 op("SqueezeV2");
  op.UpdateInputDesc("x", create_desc({1, 3, 4, 5, 1}, ge::DT_INT32));
  op.SetAttr("axis", {0, 4});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {3, 4, 5};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(SqueezeV2_ut, squeezev2_three_axis) {
  ge::op::SqueezeV2 op("SqueezeV2");
  op.UpdateInputDesc("x", create_desc({3, 4, 1, 5, 1, 1}, ge::DT_INT32));
  op.SetAttr("axis", {2, 4, 5});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {3, 4, 5};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(SqueezeV2_ut, squeezev2_unsorted_axis) {
  ge::op::SqueezeV2 op("SqueezeV2");
  op.UpdateInputDesc("x", create_desc({3, 4, 1, 5, 1, 1}, ge::DT_INT32));
  op.SetAttr("axis", {5, 4, 2});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {3, 4, 5};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

// abnormal_axis: we use input as output 
TEST_F(SqueezeV2_ut, squeezev2_abnormal_axis) {
  ge::op::SqueezeV2 op("SqueezeV2");
  op.UpdateInputDesc("x", create_desc({3, 4, 5, 1}, ge::DT_INT32));
  op.SetAttr("axis", {5});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {3, 4, 5, 1};  
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(SqueezeV2_ut, squeezev2_repetitive_axis) {
  ge::op::SqueezeV2 op("SqueezeV2");
  op.UpdateInputDesc("x", create_desc({3, 1, 4, 5}, ge::DT_INT32));
  op.SetAttr("axis", {1, 1});
  auto ret = op.InferShapeAndType();
  auto output = op.GetOutputDesc("y");
  std::vector<int64_t> output_shape = {3, 4, 5};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(SqueezeV2_ut, squeezev2_one_axis_with_range) {
  ge::op::SqueezeV2 op("SqueezeV2");
  ge::TensorDesc desc = create_desc({-1, 1, 5}, ge::DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> input_range = {{1, -1}, {1, 1}, {5, 5}};
  desc.SetShapeRange(input_range);
  op.UpdateInputDesc("x", desc);
  op.SetAttr("axis", {1});

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");

  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);

  std::vector<int64_t> output_shape = {-1, 5};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
  std::vector<std::pair<int64_t, int64_t>> expect_output_range = {{1, -1}, {5, 5}};
  std::vector<std::pair<int64_t, int64_t>>output_range;
  output.GetShapeRange(output_range);
  EXPECT_EQ(output_range, expect_output_range);
}

TEST_F(SqueezeV2_ut, squeezev2_two_axis_with_range) {
  ge::op::SqueezeV2 op("SqueezeV2");
  ge::TensorDesc desc = create_desc({-1, 1, 1, 5}, ge::DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> input_range = {{1, -1}, {1, 1}, {1, 1}, {5, 5}};
  desc.SetShapeRange(input_range);
  op.UpdateInputDesc("x", desc);
  op.SetAttr("axis", {1, 2});

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");

  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);

  std::vector<int64_t> output_shape = {-1, 5};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
  std::vector<std::pair<int64_t, int64_t>> expect_output_range = {{1, -1}, {5, 5}};
  std::vector<std::pair<int64_t, int64_t>>output_range;
  output.GetShapeRange(output_range);
  EXPECT_EQ(output_range, expect_output_range);
}

TEST_F(SqueezeV2_ut, squeezev2_unsorted_axis_with_range) {
  ge::op::SqueezeV2 op("SqueezeV2");
  ge::TensorDesc desc = create_desc({-1, -1, 1, 1, 5}, ge::DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> input_range = {{1, -1}, {1, -1}, {1, 1}, {1, 1}, {5, 5}};
  desc.SetShapeRange(input_range);
  op.UpdateInputDesc("x", desc);
  op.SetAttr("axis", {3, 2});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);

  std::vector<int64_t> output_shape = {-1, -1, 5};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);

  std::vector<std::pair<int64_t, int64_t>> expect_output_range = {{1, -1}, {1, -1}, {5, 5}};
  std::vector<std::pair<int64_t, int64_t>> output_range;
  output.GetShapeRange(output_range);
  EXPECT_EQ(output_range, expect_output_range);
}

TEST_F(SqueezeV2_ut, squeezev2_abnormal_axis_with_range) {
  ge::op::SqueezeV2 op("SqueezeV2");
  ge::TensorDesc desc = create_desc({-1, -1, 5}, ge::DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> input_range = {{1, -1}, {1, -1}, {5, 5}};
  desc.SetShapeRange(input_range);
  op.UpdateInputDesc("x", desc);
  op.SetAttr("axis", {5});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {-1, -1, 5};  
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);

  std::vector<std::pair<int64_t, int64_t>> expect_output_range = {{1, -1}, {1, -1}, {5, 5}};
  std::vector<std::pair<int64_t, int64_t>> output_range;
  output.GetShapeRange(output_range);
}

TEST_F(SqueezeV2_ut, squeezev2_repetitive_axis_with_range) {
  ge::op::SqueezeV2 op("SqueezeV2");
  ge::TensorDesc desc = create_desc({-1, -1, 5}, ge::DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> input_range = {{1, -1}, {1, -1}, {5, 5}};
  desc.SetShapeRange(input_range);
  op.UpdateInputDesc("x", desc);
  op.SetAttr("axis", {1, 1});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output = op.GetOutputDesc("y");
  std::vector<int64_t> output_shape = {-1, 5};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);

  std::vector<std::pair<int64_t, int64_t>> expect_output_range = {{1, -1}, {5, 5}};
  std::vector<std::pair<int64_t, int64_t>>output_range;
  output.GetShapeRange(output_range);
  EXPECT_EQ(output_range, expect_output_range);
}

TEST_F(SqueezeV2_ut, squeezev2_empty_axis_with_range) {
  ge::op::SqueezeV2 op("SqueezeV2");
  ge::TensorDesc desc = create_desc({-1, 1, -1, 1, 5}, ge::DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> input_range = {{1, -1}, {1, 1}, {1, -1}, {1, 1}, {5, 5}};
  desc.SetShapeRange(input_range);
  op.UpdateInputDesc("x", desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output = op.GetOutputDesc("y");
  std::vector<int64_t> output_shape = {-1, -1, 5};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);

  std::vector<std::pair<int64_t, int64_t>> expect_output_range = {{1, -1}, {1, -1}, {5, 5}};
  std::vector<std::pair<int64_t, int64_t>>output_range;
  output.GetShapeRange(output_range);
  EXPECT_EQ(output_range, expect_output_range);

}
