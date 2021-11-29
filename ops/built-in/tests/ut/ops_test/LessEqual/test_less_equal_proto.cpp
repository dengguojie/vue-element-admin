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
 * @file test_less_equal_proto.cpp
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

class less_equal : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "less_equal SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "less_equal TearDown" << std::endl;
  }
};

TEST_F(less_equal, less_equal_infer_shape_fp16) {
  ge::op::LessEqual op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {64},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 100},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(less_equal, InferShapeLessEqual_000) {
  ge::op::LessEqual op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}};
  op.UpdateInputDesc("x1",
                     create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_ND, {64}, ge::FORMAT_ND, shape_range));
  op.UpdateInputDesc("x2",
                     create_desc_shape_range({-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {64}, ge::FORMAT_ND, shape_range));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}