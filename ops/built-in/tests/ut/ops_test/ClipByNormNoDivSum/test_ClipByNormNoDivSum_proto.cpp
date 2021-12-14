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

class clip_by_norm_no_div_sum : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "clip_by_norm_no_div_sum SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "clip_by_norm_no_div_sum TearDown" << std::endl;
  }
};


TEST_F(clip_by_norm_no_div_sum, clip_by_norm_no_div_sum_infershape_diff_test_1) {
  ge::op::ClipByNormNoDivSum op;
  op.UpdateInputDesc("x", create_desc_shape_range({-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1},
                                                                    ge::FORMAT_ND, {{1, 100}}));
  op.UpdateInputDesc("greater_zeros", create_desc_shape_range({-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1},
                                                                    ge::FORMAT_ND, {{1, 100}}));
  op.UpdateInputDesc("select_ones", create_desc_shape_range({-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1},
                                                                  ge::FORMAT_ND, {{1, 100}}));
  op.UpdateInputDesc("maximum_ones", create_desc_shape_range({-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1},
                                                                   ge::FORMAT_ND, {{1, 100}}));

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  ge::TensorDesc tensor_greater_zeros = op.GetInputDesc("greater_zeros");
  ge::TensorDesc tensor_select_ones = op.GetInputDesc("select_ones");
  ge::TensorDesc tensor_maximum_ones = op.GetInputDesc("maximum_ones");

  op.UpdateInputDesc("x", tensor_x);
  op.UpdateInputDesc("greater_zeros", tensor_greater_zeros);
  op.UpdateInputDesc("select_ones", tensor_select_ones);
  op.UpdateInputDesc("maximum_ones", tensor_maximum_ones);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y1_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);

  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1,100}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(clip_by_norm_no_div_sum, InfershapeClipByNormNoDivSum_001) {
  ge::op::ClipByNormNoDivSum op;
  op.UpdateInputDesc("x", create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {1}, ge::FORMAT_ND, {{1, 100}}));
  op.UpdateInputDesc("greater_zeros",
                     create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {1}, ge::FORMAT_ND, {{1, 100}}));
  op.UpdateInputDesc("select_ones",
                     create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {1}, ge::FORMAT_ND, {{1, 100}}));
  op.UpdateInputDesc("maximum_ones",
                     create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {1}, ge::FORMAT_ND, {{1, 100}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}