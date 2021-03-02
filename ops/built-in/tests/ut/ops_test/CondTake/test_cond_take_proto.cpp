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
 * @file test_fifo_queue_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "condtake_ops.h"

class CondTakeTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CondTake SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CondTake TearDown" << std::endl;
  }
};

TEST_F(CondTakeTest, InferShape) {
  ge::op::CondTake op;
  op.UpdateInputDesc("data", create_desc({2, 2}, ge::DT_FLOAT));
  op.UpdateInputDesc("mask", create_desc({2, 2}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_data_desc = op.GetOutputDesc("out_data");
  EXPECT_EQ(out_data_desc.GetDataType(), ge::DT_FLOAT); 
  std::vector<int64_t> expected_sum_indices_shape = {2, 2};
  EXPECT_EQ(out_data_desc.GetShape().GetDims(), expected_sum_indices_shape);

  auto out_index_desc = op.GetOutputDesc("out_index");
  EXPECT_EQ(out_index_desc.GetDataType(), ge::DT_INT32); 
  std::vector<int64_t> expected_sum_values_shape = {2, 2};
  EXPECT_EQ(out_index_desc.GetShape().GetDims(), expected_sum_values_shape);

  auto valid_num_desc = op.GetOutputDesc("valid_num");
  EXPECT_EQ(valid_num_desc.GetDataType(), ge::DT_INT32); 
  std::vector<int64_t> expected_sum_shape_shape = {1};
  EXPECT_EQ(valid_num_desc.GetShape().GetDims(), expected_sum_shape_shape);
}

