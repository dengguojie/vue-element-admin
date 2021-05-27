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
 * @file test_sparse_dense_cwise_mul_proto.cpp
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
class SparseDenseCwiseMulTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseDenseCwiseMul SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseDenseCwiseMul TearDown" << std::endl;
  }
};

TEST_F(SparseDenseCwiseMulTest, InferShape_01) {
  ge::op::SparseDenseCwiseMul op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  op.UpdateInputDesc("x1_indices", create_desc({2, 3}, ge::DT_INT64));
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}