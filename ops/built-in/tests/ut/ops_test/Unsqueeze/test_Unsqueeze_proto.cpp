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
 * @file test_Unsqueeze_proto.cpp
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

class Unsqueeze_ut : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Unsqueeze_ut SetUp" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "Unsqueeze_ut TearDown" << std::endl;
  }
};

TEST_F(Unsqueeze_ut, unsqueeze_negative_axes) {
  ge::op::Unsqueeze op("Unsqueeze");
  op.UpdateInputDesc("x", create_desc({1, 3, 2, 5}, ge::DT_INT32));
  op.SetAttr("axes", {-5});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {1, 1, 3, 2, 5};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(Unsqueeze_ut, unsqueeze_one_axis) {
  ge::op::Unsqueeze op("Unsqueeze");
  op.UpdateInputDesc("x", create_desc({3, 4, 5}, ge::DT_INT32));
  op.SetAttr("axes", {1});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {3, 1, 4, 5};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(Unsqueeze_ut, unsqueeze_two_axes) {
  ge::op::Unsqueeze op("Unsqueeze");
  op.UpdateInputDesc("x", create_desc({3, 4, 5}, ge::DT_INT32));
  op.SetAttr("axes", {0, 4});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {1, 3, 4, 5, 1};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(Unsqueeze_ut, unsqueeze_three_axes) {
  ge::op::Unsqueeze op("Unsqueeze");
  op.UpdateInputDesc("x", create_desc({3, 4, 5}, ge::DT_INT32));
  op.SetAttr("axes", {2, 4, 5});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {3, 4, 1, 5, 1, 1};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(Unsqueeze_ut, unsqueeze_unsorted_axes) {
  ge::op::Unsqueeze op("Unsqueeze");
  op.UpdateInputDesc("x", create_desc({3, 4, 5}, ge::DT_INT32));
  op.SetAttr("axes", {5, 4, 2});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {3, 4, 1, 5, 1, 1};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(Unsqueeze_ut, unsqueeze_abnormal_axes) {
  ge::op::Unsqueeze op("Unsqueeze");
  op.UpdateInputDesc("x", create_desc({3, 4, 5}, ge::DT_INT32));
  op.SetAttr("axes", {4});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Unsqueeze_ut, unsqueeze_repetitive_axes) {
  ge::op::Unsqueeze op("Unsqueeze");
  op.UpdateInputDesc("x", create_desc({3, 4, 5}, ge::DT_INT32));
  op.SetAttr("axes", {1, 1});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}