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
 * @file test_dynamic_AddN_proto.cpp
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

class AddN : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AddN SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AddN TearDown" << std::endl;
  }
};

TEST_F(AddN, add_n_case_0) {
  ge::op::AddN op;
  std::vector<std::pair<int64_t,int64_t>> shape_range_0 = {{2, 6}, {6, 6}, {1, 10}};
  std::vector<std::pair<int64_t,int64_t>> shape_range_1 = {{4, 4}, {2, 4}, {4, 8}};
  auto tensor_desc_0 =
    create_desc_shape_range({-1, 6, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {-1, 6, -1}, ge::FORMAT_ND, shape_range_0);
  auto tensor_desc_1 =
    create_desc_shape_range({4, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, -1, -1}, ge::FORMAT_ND, shape_range_1);

  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc_0);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc_1);
  op.SetAttr("N", 2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4, 6, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {4, 4},
      {6, 6},
      {4, 8},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}


TEST_F(AddN, add_n_case_1) {
  ge::op::AddN op;
  std::vector<std::pair<int64_t,int64_t>> shape_range_0 = {{2, 6}, {6, 6}, {1, 10}};
  std::vector<std::pair<int64_t,int64_t>> shape_range_1 = {{4, 4}, {2, 4}, {4, 8}};
  auto tensor_desc_0 =
    create_desc_shape_range({-1, 6, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {-1, 6, -1}, ge::FORMAT_ND, shape_range_0);
  auto tensor_desc_1 =
    create_desc_shape_range({4, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, -1, -1}, ge::FORMAT_ND, shape_range_1);
  auto tensor_desc_2 =
      create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {-2}, ge::FORMAT_ND, shape_range_1);

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc_0);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc_1);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc_2);
  op.SetAttr("N", 3);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4, 6, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {4, 4},
      {6, 6},
      {4, 8},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}


TEST_F(AddN, add_n_case_2) {
  ge::op::AddN op;
  std::vector<std::pair<int64_t,int64_t>> shape_range_0 = {{1,11}, {1,11}, {1,11},{1,11},{1,11},{1,11},{1,11}};
  std::vector<std::pair<int64_t,int64_t>> shape_range_1 = {{1,12}, {1,12}, {1,12},{1,12},{1,12},{1,12},{1,12}};
  auto tensor_desc_0 =
    create_desc_shape_range({-1, -1,-1,-1,-1,-1,-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {-1, -1,-1,-1,-1,-1,-1},
    ge::FORMAT_ND, shape_range_0);
  auto tensor_desc_1 =
    create_desc_shape_range({-1, -1,-1,-1,-1,-1,-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {-1, -1,-1,-1,-1,-1,-1},
    ge::FORMAT_ND, shape_range_1);

  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc_0);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc_1);
  op.SetAttr("N", 2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1,-1,-1,-1,-1,-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {1, 11},
      {1, 11},
      {1, 11},
      {1, 11},
      {1, 11},
      {1, 11},
      {1, 11}
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}


TEST_F(AddN, add_n_case_3) {
  ge::op::AddN op;
  std::vector<std::pair<int64_t,int64_t>> shape_range_0 = {{2,3}, {2,3}};
  std::vector<std::pair<int64_t,int64_t>> shape_range_1 = {{1,-1}, {1,-1}};
  auto tensor_desc_0 =
    create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {-1, -1}, ge::FORMAT_ND, shape_range_0);
  auto tensor_desc_1 =
    create_desc_shape_range({-1,-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {-1, -1}, ge::FORMAT_ND, shape_range_1);

  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc_0);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc_1);
  op.SetAttr("N", 2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 3},
      {2, 3}
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}


TEST_F(AddN, add_n_case_4) {
  ge::op::AddN op;
  std::vector<std::pair<int64_t,int64_t>> shape_range_1 = {{8, 8}, {1, 10}};
  std::vector<std::pair<int64_t,int64_t>> shape_range_2 = {{8, 8}, {3, 3}};
  auto tensor_desc_0 =
    create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {-2}, ge::FORMAT_ND, shape_range_1);
  auto tensor_desc_1 =
    create_desc_shape_range({8,-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {8,-1}, ge::FORMAT_ND, shape_range_1);
  auto tensor_desc_2 =
      create_desc_shape_range({8,3}, ge::DT_FLOAT16, ge::FORMAT_ND, {8,3}, ge::FORMAT_ND, shape_range_2);

  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc_0);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc_1);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc_2);
  op.SetAttr("N", 3);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {8,3};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {8, 8},
      {3, 3},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}


TEST_F(AddN, add_n_case_5) {
  ge::op::AddN op;
  std::vector<std::pair<int64_t,int64_t>> shape_range_0 = {{1, 10}, {1, 10}};
  std::vector<std::pair<int64_t,int64_t>> shape_range_1 = {{1, 20}, {3, 3}};
  auto tensor_desc_0 =
    create_desc_shape_range({-1,-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {-1,-1}, ge::FORMAT_ND, shape_range_0);
  auto tensor_desc_1 =
    create_desc_shape_range({-1,3}, ge::DT_FLOAT16, ge::FORMAT_ND, {-1,3}, ge::FORMAT_ND, shape_range_1);


  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc_0);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc_1);
  op.SetAttr("N", 2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1,3};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {1, 10},
      {3, 3},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}