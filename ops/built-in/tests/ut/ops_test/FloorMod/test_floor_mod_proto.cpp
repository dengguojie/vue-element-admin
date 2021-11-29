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
 * @file test_floor_mod_proto.cpp
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

class floor_mod : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "floor_mod SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "floor_mod TearDown" << std::endl;
  }
};

TEST_F(floor_mod, floor_mod_infershape_diff_test_4) {
  ge::op::FloorMod op;
  op.UpdateInputDesc("x1", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND, {{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("x2", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND, {{1,6},{4,7}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(floor_mod, floor_mod_infershape_diff_test_5) {
  ge::op::FloorMod op;
  op.UpdateInputDesc("x1", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND, {{1,10}, {1,10},{2,6}}));
  op.UpdateInputDesc("x2", create_desc_shape_range({10, -1}, ge::DT_INT32, ge::FORMAT_ND, {10, -1}, ge::FORMAT_ND, {{10,10}, {1,7}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(floor_mod, floor_mod_infershape_diff_test_6) {
  ge::op::FloorMod op;
  op.UpdateInputDesc("x1", create_desc_shape_range({3, 4, 5, 6, -1}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, -1}, ge::FORMAT_ND, {{3,3},{4,4},{5,5},{6,6},{3,8}}));
  op.UpdateInputDesc("x2", create_desc_shape_range({3, 4, 5, 6, 1}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 1}, ge::FORMAT_ND, {{3,3},{4,4},{5,5},{6,6},{1,1}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {3, 4, 5, 6, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{3,3},{4,4},{5,5},{6,6},{3,8}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(floor_mod, floor_mod_infershape_diff_test_7) {
  ge::op::FloorMod op;
  op.UpdateInputDesc("x1", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND, {{1,10}, {2,7},{2,6}}));
  op.UpdateInputDesc("x2", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND, {{2,7}, {1,7}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(floor_mod, floor_mod_infershape_diff_test_8) {
  ge::op::FloorMod op;
  op.UpdateInputDesc("x1", create_desc_shape_range({-1,2}, ge::DT_INT32, ge::FORMAT_ND, {-1,2}, ge::FORMAT_ND, {{1,8}, {2,2}}));
  op.UpdateInputDesc("x2", create_desc_shape_range({1}, ge::DT_INT32, ge::FORMAT_ND, {1}, ge::FORMAT_ND, {}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-1,2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,8}, {2,2}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(floor_mod, floor_mod_infershape_diff_test_9) {
  ge::op::FloorMod op;
  op.UpdateInputDesc("x1", create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {-1}, ge::FORMAT_ND, {{1,55}}));
  op.UpdateInputDesc("x2", create_desc_shape_range({17, 2, 5, 1}, ge::DT_INT32, ge::FORMAT_ND, {17, 2, 5, 1}, ge::FORMAT_ND, {{17,17},{2,2},{5,5},{1,1}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {17, 2, 5, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{17,17},{2,2},{5,5},{1,55}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(floor_mod, floor_mod_infershape_diff_test_10) {
  ge::op::FloorMod op;
  op.UpdateInputDesc("x1", create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {-1}, ge::FORMAT_ND, {{1,55}}));
  op.UpdateInputDesc("x2", create_desc_shape_range({17, 2, 5, 1}, ge::DT_INT32, ge::FORMAT_ND, {17, 2, 5, 1}, ge::FORMAT_ND, {}));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {17, 2, 5, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{17,17},{2,2},{5,5},{1,55}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}


TEST_F(floor_mod, InferShapeFloorMod_001) {
  ge::op::FloorMod op;
  op.UpdateInputDesc("x1", create_desc_shape_range({-1}, ge::DT_INT16, ge::FORMAT_ND, {-1}, ge::FORMAT_ND, {{1,55}}));
  op.UpdateInputDesc("x2", create_desc_shape_range({17, 2, 5, 1}, ge::DT_INT32, ge::FORMAT_ND, {17, 2, 5, 1}, ge::FORMAT_ND, {}));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}