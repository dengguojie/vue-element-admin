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
 * @file test_combinations_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "split_combination_ops.h"

class combinations : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "combinations SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "combinations TearDown" << std::endl;
  }
};

TEST_F(combinations, combinations_infer_shape_int32) {
  ge::op::Combinations op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({4}, ge::DT_INT32, ge::FORMAT_ND,
                                          {4}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("r", 2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {6, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(combinations, combinations_infer_shape_fp16) {
  ge::op::Combinations op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({4}, ge::DT_FLOAT16, ge::FORMAT_ND,
                                          {4}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("r", 2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {6, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}
