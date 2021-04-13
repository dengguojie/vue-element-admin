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
#include "data_flow_ops.h"

class ResourceAccumulatorTakeGradientTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ResourceAccumulatorTakeGradientTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ResourceAccumulatorTakeGradientTest TearDown" << std::endl;
  }
};

TEST_F(ResourceAccumulatorTakeGradientTest, InferShape_01) {
  ge::op::ResourceAccumulatorTakeGradient op;
  op.UpdateInputDesc("num_required", create_desc({}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_DOUBLE);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ResourceAccumulatorTakeGradientTest, InferShape_02) {
  ge::op::ResourceAccumulatorTakeGradient op;
  op.UpdateInputDesc("num_required", create_desc({2}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_DOUBLE);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResourceAccumulatorTakeGradientTest, InferShape_03) {
  ge::op::ResourceAccumulatorTakeGradient op;
  op.UpdateInputDesc("num_required", create_desc({}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}