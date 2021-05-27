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
 * @file test_sparse_add_grad_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "sparse_ops.h"
class SparseAddGradTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseAddGrad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseAddGrad TearDown" << std::endl;
  }
};

TEST_F(SparseAddGradTest, InferShape_01) {
  ge::op::SparseAddGrad op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  op.UpdateInputDesc("backprop_val_grad", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("x1_indices", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("x2_indices", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("sum_indices", create_desc({2, 3}, ge::DT_INT64));
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  op.UpdateInputDesc("x2_indices", create_desc({2, 3}, ge::DT_INT64));
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}