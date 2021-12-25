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
 * @file test_exponential_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "random_ops.h"
#include "split_combination_ops.h"

class Exponential : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Exponential SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Exponential TearDown" << std::endl;
  }
};

TEST_F(Exponential, exponential_infer_shape_float16) {
  ge::op::Exponential op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc_shape_x = create_desc_shape_range({2,2},
                                                  ge::DT_FLOAT16, ge::FORMAT_ND,
                                                  {2,2},
                                                  ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc_shape_x);
  op.SetAttr("lambda", ge::DT_FLOAT16);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2,2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(Exponential, exponential_infer_shape_float32) {
  ge::op::Exponential op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc_shape_x = create_desc_shape_range({2,2},
                                                  ge::DT_FLOAT, ge::FORMAT_ND,
                                                  {2,2},
                                                  ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc_shape_x);
  op.SetAttr("lambda", ge::DT_FLOAT);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {2,2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(Exponential, exponential_infer_shape_float64) {
  ge::op::Exponential op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc_shape_x = create_desc_shape_range({2,2},
                                                  ge::DT_DOUBLE, ge::FORMAT_ND,
                                                  {2,2},
                                                  ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc_shape_x);
  op.SetAttr("lambda", ge::DT_DOUBLE);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_DOUBLE);
  std::vector<int64_t> expected_output_shape = {2,2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
