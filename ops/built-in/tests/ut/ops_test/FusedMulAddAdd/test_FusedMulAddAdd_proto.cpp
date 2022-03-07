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
 * @file test_FusedMulAddAdd.cpp
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

class FusedMulAddAdd : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FusedMulAddAdd SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FusedMulAddAdd TearDown" << std::endl;
  }
};

TEST_F(FusedMulAddAdd, FusedMulAddAdd_infershape_test_1) {
  ge::op::FusedMulAddAdd op;

  op.UpdateInputDesc("x1", create_desc_shape_range({289, 40, 16, 16}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, {289, 40, 16, 16}, ge::FORMAT_FRACTAL_NZ, {{1,1000}, {1,1000}, {1,1000}, {1,1000}}));
  op.UpdateInputDesc("x2", create_desc_shape_range({640, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {640, 1}, ge::FORMAT_ND, {{1,1000}, {1,1000}}));
  op.UpdateInputDesc("x3", create_desc_shape_range({4624}, ge::DT_FLOAT, ge::FORMAT_ND, {4624}, ge::FORMAT_ND, {{1,1000}}));
  op.UpdateInputDesc("x4", create_desc_shape_range({289, 40, 16, 16}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, {289, 40, 16, 16}, ge::FORMAT_FRACTAL_NZ, {{1,1000}, {1,1000}, {1,1000}, {1,1000}}));

  ge::TensorDesc tensor_x1 = op.GetInputDesc("x1");
  ge::TensorDesc tensor_x2 = op.GetInputDesc("x2");
  ge::TensorDesc tensor_x3 = op.GetInputDesc("x3");
  ge::TensorDesc tensor_x4 = op.GetInputDesc("x4");

  op.UpdateInputDesc("x1", tensor_x1);
  op.UpdateInputDesc("x2", tensor_x2);
  op.UpdateInputDesc("x3", tensor_x3);
  op.UpdateInputDesc("x4", tensor_x4);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y1_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_output_shape = {289, 40, 16, 16};
  EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(FusedMulAddAdd, FusedMulAddAdd_infershape_test_2) {
  ge::op::FusedMulAddAdd op;

  op.UpdateInputDesc("x1",
                     create_desc_shape_range({1}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, {1}, ge::FORMAT_FRACTAL_NZ, {{1, 1}}));
  op.UpdateInputDesc("x2",
                     create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::FORMAT_ND, {{1, 3}}));
  op.UpdateInputDesc("x3",
                     create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::FORMAT_ND, {{1, 3}}));
  op.UpdateInputDesc("x4",
                     create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, {1}, ge::FORMAT_FRACTAL_NZ, {{1, 1}}));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FusedMulAddAdd, FusedMulAddAdd_infershape_test_3) {
  ge::op::FusedMulAddAdd op;

  op.UpdateInputDesc("x1",
                     create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, {1}, ge::FORMAT_FRACTAL_NZ, {{1, 1}}));
  op.UpdateInputDesc("x2",
                     create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::FORMAT_ND, {{1, 3}}));
  op.UpdateInputDesc("x3",
                     create_desc_shape_range({1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1}, ge::FORMAT_ND, {{1, 3}}));
  op.UpdateInputDesc("x4",
                     create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, {1}, ge::FORMAT_FRACTAL_NZ, {{1, 1}}));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FusedMulAddAdd, FusedMulAddAdd_infershape_test_4) {
  ge::op::FusedMulAddAdd op;

  op.UpdateInputDesc("x1",
                     create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, {1}, ge::FORMAT_FRACTAL_NZ, {{1, 1}}));
  op.UpdateInputDesc("x2",
                     create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::FORMAT_ND, {{1, 3}}));
  op.UpdateInputDesc("x3",
                     create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::FORMAT_ND, {{1, 3}}));
  op.UpdateInputDesc("x4",
                     create_desc_shape_range({1}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, {1}, ge::FORMAT_FRACTAL_NZ, {{1, 1}}));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FusedMulAddAdd, FusedMulAddAdd_infershape_test_5) {
  ge::op::FusedMulAddAdd op;

  op.UpdateInputDesc("x1", create_desc_shape_range({-1, 40, 16, 16}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, {289, 40, 16, 16}, ge::FORMAT_FRACTAL_NZ, {{-1,1000}, {1,1000}, {1,1000}, {1,1000}}));
  op.UpdateInputDesc("x2", create_desc_shape_range({-1, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {640, 1}, ge::FORMAT_ND, {{-1,1000}, {1,1000}}));
  op.UpdateInputDesc("x3", create_desc_shape_range({4624}, ge::DT_FLOAT, ge::FORMAT_ND, {4624}, ge::FORMAT_ND, {{1,10000}}));
  op.UpdateInputDesc("x4", create_desc_shape_range({-1, 40, 16, 16}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, {289, 40, 16, 16}, ge::FORMAT_FRACTAL_NZ, {{-1,1000}, {1,1000}, {1,1000}, {1,1000}}));

  ge::TensorDesc tensor_x1 = op.GetInputDesc("x1");
  ge::TensorDesc tensor_x2 = op.GetInputDesc("x2");
  ge::TensorDesc tensor_x3 = op.GetInputDesc("x3");
  ge::TensorDesc tensor_x4 = op.GetInputDesc("x4");
  op.UpdateInputDesc("x1", tensor_x1);
  op.UpdateInputDesc("x2", tensor_x2);
  op.UpdateInputDesc("x3", tensor_x3);
  op.UpdateInputDesc("x4", tensor_x4);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}