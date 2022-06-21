/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this
 file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_tril_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "matrix_calculation_ops.h"
#include "op_proto_test_util.h"

class tril : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "tril SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "tril TearDown" << std::endl; }
};

TEST_F(tril, tril_infer_test_1) {
  ge::op::Tril op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 10}, ge::DT_FLOAT16, ge::FORMAT_ND, {},
                                ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {6, 10};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(tril, tril_infer_test_2) {
  ge::op::Tril op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({7, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {},
                                ge::FORMAT_ND));
  int diagonal = 1;
  op.SetAttr("diagonal", diagonal);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {7, 5};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
