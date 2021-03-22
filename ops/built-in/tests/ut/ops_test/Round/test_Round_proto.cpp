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
 * @file test_round_proto.cpp
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

class round : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "round SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "round TearDown" << std::endl;
  }
};

TEST_F(round, round_infershape_diff_test) {
  ge::op::Round op;
  op.UpdateInputDesc("x", create_desc({4, 3, 4}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetOriginShape().GetDims(), expected_output_shape);
}

TEST_F(round, round_dynamic_infershape_test_01) {
  ge::op::Round op;
  std::vector<std::pair<int64_t, int64_t>> shape_range ={{1, 1}, {1, 16}, {1, 16}};
  auto tensor_desc = create_desc_shape_range({1, -1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {1, 16, 16},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, -1, -1};
  std::vector<int64_t> expected_origin_shape = {1, 16, 16};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {
    {1, 1},
    {1, 16},
    {1, 16}
  };
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetOriginShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

