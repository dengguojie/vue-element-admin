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
 * @file test_relu_grad_v2_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

class relu_grad_v2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "relu_grad_v2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "relu_grad_v2 TearDown" << std::endl;
  }
};

TEST_F(relu_grad_v2, relu_grad_v2_infer_shape_01) {
  ge::op::ReluGradV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {1, 1}, {10, 10}, {11, 11}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, -1, -1, -1, 16},
                                             ge::DT_FLOAT16, ge::FORMAT_NHWC,
                                             {2, 10, 11, 16},
                                             ge::FORMAT_NHWC, shape_range);
  auto mask_desc = create_desc_shape_range({-1, -1, -1, -1, 16},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 1, 10, 11, 16},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("gradients", tensor_desc);
  op.UpdateInputDesc("mask", mask_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("backprops");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1, 16};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {
    {2, 2},
    {1, 1},
    {10, 10},
    {11, 11},
    {16, 16}
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
