/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_range_d_proto.cpp
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

class RangeD : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RangeD SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RangeD TearDown" << std::endl;
  }
};

TEST_F(RangeD, Range_infer_shape_0) {
  ge::op::RangeD op;
  float start = 1, limit = 2, delta = 3;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1, 1, 1}, ge::DT_INT32, ge::FORMAT_NHWC, {1, 1, 1,1}, ge::FORMAT_NHWC));
  op.SetAttr("start", start);
  op.SetAttr("limit", limit);
  op.SetAttr("delta", delta);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RangeD, Range_infer_shape_1) {
  ge::op::RangeD op;
  float start = 1, limit = 2, delta = 3;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1, 1, 1,1}, ge::FORMAT_NHWC));
  op.SetAttr("start", start);
  op.SetAttr("limit", limit);
  op.SetAttr("delta", delta);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RangeD, Range_infer_shape_2) {
  ge::op::RangeD op;
  float start = 1, limit = 2, delta = 3;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1, 1, 1,1}, ge::FORMAT_NHWC));
  op.SetAttr("limit", limit);
  op.SetAttr("delta", delta);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RangeD, Range_infer_shape_3) {
  ge::op::RangeD op;
  float start = 1, limit = 2, delta = 3;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1, 1, 1,1}, ge::FORMAT_NHWC));
  op.SetAttr("start", start);
  op.SetAttr("delta", delta);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RangeD, Range_infer_shape_4) {
  ge::op::RangeD op;
  float start = 1, limit = 2, delta = 3;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1, 1, 1,1}, ge::FORMAT_NHWC));
  op.SetAttr("start", start);
  op.SetAttr("limit", limit);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
