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

class Bias : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "bias SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "bias TearDown" << std::endl;
  }
};

TEST_F(Bias, bias_infershape_test) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 10},{3,10},{4,10}};
  auto tensor_desc_x = create_desc_shape_range({-1,-1,-1},
                                               ge::DT_FLOAT16, ge::FORMAT_ND,
                                               {2,3,4},
                                               ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("axis", 0);
  op.SetAttr("num_axes", 3);
  op.SetAttr("bias_from_blob", true);
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1,-1,-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
    {2, 10},{3,10},{4,10}
  };
  
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(Bias, InfershapeBias_test_001) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 10}, {3, 10}, {4, 10}};
  auto tensor_desc_x =
      create_desc_shape_range({-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("axis", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(Bias, InfershapeBias_test_002) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 10}, {3, 10}, {4, 10}};
  auto tensor_desc_x =
      create_desc_shape_range({-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("num_axes", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(Bias, InfershapeBias_test_003) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 10}, {3, 10}, {4, 10}};
  auto tensor_desc_x =
      create_desc_shape_range({-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("bias", 10);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(Bias, InfershapeBias_test_004) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 10}, {3, 10}, {4, 10}};
  auto tensor_desc_x =
      create_desc_shape_range({-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("num_axes", -6);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(Bias, InfershapeBias_test_005) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 10}, {3, 10}, {4, 10}};
  auto tensor_desc_x =
      create_desc_shape_range({-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("bias_from_blob", true);
  op.SetAttr("axis", 1);
  op.SetAttr("num_axes", -1);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(Bias, InfershapeBias_test_006) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 10}, {3, 10}, {4, 10}};
  auto tensor_desc_x =
      create_desc_shape_range({-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("bias_from_blob", true);
  op.SetAttr("axis", 1);
  op.SetAttr("num_axes", 0);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(Bias, InfershapeBias_test_007) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 10}, {3, 10}, {4, 10}};
  auto tensor_desc_x =
      create_desc_shape_range({-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("bias_from_blob", true);
  op.SetAttr("axis", 4);
  op.SetAttr("num_axes", 1);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(Bias, InfershapeBias_test_008) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 10}, {3, 10}, {4, 10}};
  auto tensor_desc_x =
      create_desc_shape_range({-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("bias_from_blob", true);
  op.SetAttr("axis", 0);
  op.SetAttr("num_axes", 2);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(Bias, InfershapeBias_test_009) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 10}, {3, 10}, {4, 10}};
  auto tensor_desc_x =
      create_desc_shape_range({-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("bias_from_blob", false);
  op.SetAttr("axis", 5);
  op.SetAttr("num_axes", 2);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(Bias, InfershapeBias_test_010) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 10}, {3, 10}, {4, 10}, {5, 10}};
  auto tensor_desc_x = create_desc_shape_range({-1, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4, 5},
                                               ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("axis", 0);
  op.SetAttr("num_axes", -1);
  op.SetAttr("bias_from_blob", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{2, 10}, {3, 10}, {4, 10}, {5, 10}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(Bias, InfershapeBias_test_011) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 10}, {3, 10}, {4, 10}, {5, 10}};
  auto tensor_desc_x = create_desc_shape_range({-1, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4, 5},
                                               ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("axis", 0);
  op.SetAttr("num_axes", 2);
  op.SetAttr("bias_from_blob", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(Bias, InfershapeBias_test_012) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 10}, {3, 10}, {4, 10}, {5, 10}};
  auto tensor_desc_x = create_desc_shape_range({-1, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4, 5},
                                               ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("bias_from_blob", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(Bias, InfershapeBias_test_013) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 10}, {3, 10}, {4, 10}};
  auto tensor_desc_x =
      create_desc_shape_range({-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("bias_from_blob", {});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(Bias, InfershapeBias_test_014) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 10}, {3, 10}, {4, 10}};
  auto tensor_desc_x =
      create_desc_shape_range({-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("bias_from_blob", true);
  op.SetAttr("num_axes", 12);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(Bias, InfershapeBias_test_015) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 10}, {3, 10}, {4, 10}};
  auto tensor_desc_x =
      create_desc_shape_range({-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 4}, ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("bias_from_blob", false);
  op.SetAttr("num_axes", 12);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(Bias, InfershapeBias_test_016) {
  ge::op::Bias op;
  auto tensor_desc_x = create_desc_with_ori({4, 3, 2, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3, 2, 4}, ge::FORMAT_ND);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("axis", -1);
  op.SetAttr("num_axes", -1);
  op.SetAttr("bias_from_blob", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(Bias, InfershapeBias_test_017) {
  ge::op::Bias op;
  auto tensor_desc_x = create_desc_with_ori({4, 3, 2, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3, 2, 4}, ge::FORMAT_ND);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("axis", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Bias, InfershapeBias_test_018) {
  ge::op::Bias op;
  auto tensor_desc_x = create_desc_with_ori({4, 3, 2, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3, 2, 4}, ge::FORMAT_ND);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("num_axes", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Bias, InfershapeBias_test_019) {
  ge::op::Bias op;
  auto tensor_desc_x = create_desc_with_ori({4, 3, 2, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3, 2, 4}, ge::FORMAT_ND);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("bias_from_blob", {});

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}