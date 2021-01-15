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
 * @file test_mul_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class BNTrainingReduce : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BNTrainingReduce SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BNTrainingReduce TearDown" << std::endl;
  }
};

TEST_F(BNTrainingReduce, bn_training_reduce_test_1) {
  ge::op::BNTrainingReduce op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 3}, {1, 2}, {4, 18},{4, 18},{4, 18},{4, 18}};
  auto tensor_desc = create_desc_shape_range({2,1,4,5,6,16},
                                             ge::DT_FLOAT, ge::FORMAT_NDC1HWC0,
                                             {2,1,4,5,6,16},
                                             ge::FORMAT_NDC1HWC0, shape_range);
  op.UpdateInputDesc("x", tensor_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("sum");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {1,1,4,1,1,16};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
TEST_F(BNTrainingReduce, bn_training_reduce_test_2) {
  ge::op::BNTrainingReduce op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 3}, {1, 2}, {4, 18},{4, 18},{4, 18}};
  auto tensor_desc = create_desc_shape_range({2,1,4,5,6},
                                             ge::DT_FLOAT, ge::FORMAT_NCDHW,
                                             {2,1,4,5,6},
                                             ge::FORMAT_NCDHW, shape_range);
  op.UpdateInputDesc("x", tensor_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("sum");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
TEST_F(BNTrainingReduce, bn_training_reduce_test_3) {
  ge::op::BNTrainingReduce op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 3}, {1, 2}, {4, 18},{4, 18},{4, 18}};
  auto tensor_desc = create_desc_shape_range({2,1,4,5,6},
                                             ge::DT_FLOAT, ge::FORMAT_NDHWC,
                                             {2,1,4,5,6},
                                             ge::FORMAT_NDHWC, shape_range);
  op.UpdateInputDesc("x", tensor_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("sum");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {6};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}