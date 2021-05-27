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
 * @file test_sparse_accumulator_take_gradient_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
class SparseAccumulatorTakeGradientTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseAccumulatorTakeGradient SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseAccumulatorTakeGradient TearDown" << std::endl;
  }
};

TEST_F(SparseAccumulatorTakeGradientTest, InferShape_01) {
  ge::op::SparseAccumulatorTakeGradient op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  op.SetAttr("dtype", ge::DT_INT64);
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  op.UpdateInputDesc("num_required", create_desc({1}, ge::DT_INT64));
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
