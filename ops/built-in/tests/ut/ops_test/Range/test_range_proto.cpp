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
 * @file test_range_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class Range : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Range SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Range TearDown" << std::endl;
  }
};

TEST_F(Range, Range_infer_shape_0) {
  ge::op::Range op;
  op.UpdateInputDesc("start", create_desc_with_ori({1, 1, 1, 1}, ge::DT_INT32, ge::FORMAT_NHWC, {1, 1, 1,1}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("limit", create_desc_with_ori({2, 2}, ge::DT_INT32, ge::FORMAT_NHWC, {2, 2}, ge::FORMAT_NHWC));
  op.UpdateOutputDesc("delta", create_desc_with_ori({1, 1, 1,1}, ge::DT_FLOAT, ge::FORMAT_NDHWC,{1, 1, 1,1}, ge::FORMAT_NDHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(Range, Range_infer_shape_1) {
  ge::op::Range op;
  op.UpdateInputDesc("start", create_desc_with_ori({1, 1, 1, 1}, ge::DT_INT32, ge::FORMAT_NHWC, {1, 1, 1,1}, ge::FORMAT_NHWC));
  op.UpdateOutputDesc("delta", create_desc_with_ori({1, 1, 1,1}, ge::DT_FLOAT, ge::FORMAT_NDHWC,{1, 1, 1,1}, ge::FORMAT_NDHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}