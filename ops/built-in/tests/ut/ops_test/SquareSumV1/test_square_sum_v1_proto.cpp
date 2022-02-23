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
 * @file test_square_sum_v1_proto.cpp
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

class SquareSumV1 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "square_sum_v1_proto Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "square_sum_v1_proto Proto Test TearDown" << std::endl;
  }
};

TEST_F(SquareSumV1, square_sum_v1_infershape_test_01) {
  ge::op::SquareSumV1 op;

  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 2}, {1, 7}, {1, 16}};

  auto tensor_desc = create_desc_shape_range({-1, -1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 7, 16},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);

  std::vector<int64_t> axes = {0, 1};
  bool keep_dims = false;
  op.SetAttr("axis", axes);
  op.SetAttr("keep_dims", keep_dims);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetOriginShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1, 16}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(SquareSumV1, square_sum_v1_infershape_test_02) {
  ge::op::SquareSumV1 op;

  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 16}, {8, 8}, {1, 8}, {1, 32}};

  auto tensor_desc = create_desc_shape_range({-1, 8, -1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {16, 8, 8, 16},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);

  std::vector<int64_t> axes = {0, 2};
  bool keep_dims = true;
  op.SetAttr("axis", axes);
  op.SetAttr("keep_dims", keep_dims);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {1, 8, 1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetOriginShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1, 1}, {8, 8}, {1, 1}, {1, 32}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(SquareSumV1, static_square_sum_v1_infershape_test) {
  ge::op::SquareSumV1 op;

  op.UpdateInputDesc("x", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  std::vector<int64_t> axes = {0, 1};
  bool keep_dims = true;
  op.SetAttr("axis", axes);
  op.SetAttr("keep_dims", keep_dims);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_output_shape = {1, 1, 16, 32};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetOriginShape().GetDims(), expected_output_shape);
}

TEST_F(SquareSumV1, InfershapeSquareSumV1_001) {
  ge::op::SquareSumV1 op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_FLOAT16));
  op.SetAttr("axis", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SquareSumV1, InfershapeSquareSumV1_002) {
  ge::op::SquareSumV1 op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_FLOAT16));
  std::vector<int64_t> axis = {2, 4};
  op.SetAttr("axis", axis);
  op.SetAttr("keep_dims", axis);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SquareSumV1, InfershapeSquareSumV1_003) {
  ge::op::SquareSumV1 op;
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  std::vector<int64_t> axis = {};
  op.SetAttr("axis", axis);
  op.SetAttr("keep_dims", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SquareSumV1, InfershapeSquareSumV1_004) {
  ge::op::SquareSumV1 op;
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  std::vector<int64_t> axis = {-1, 2, 1};
  op.SetAttr("axis", axis);
  op.SetAttr("keep_dims", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}