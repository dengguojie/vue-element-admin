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
 * @file test_resize_bi_cubic_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "image_ops.h"

class ResizeBicubic : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ResizeBicubic SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ResizeBicubic TearDown" << std::endl;
  }
};

TEST_F(ResizeBicubic, ResizeBicubic_infer_shape) {
  ge::op::ResizeBicubic op;
  op.UpdateInputDesc("images", create_desc_with_ori({1, 1, 1, 1}, ge::DT_INT32, ge::FORMAT_NHWC, {1, 1, 1,1}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("size", create_desc_with_ori({2, 2}, ge::DT_INT32, ge::FORMAT_NHWC, {2, 2}, ge::FORMAT_NHWC));
  op.UpdateOutputDesc("y", create_desc_with_ori({1, 1, 1,1}, ge::DT_FLOAT, ge::FORMAT_NDHWC,{1, 1, 1,1}, ge::FORMAT_NDHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeBicubic, ResizeBicubic_infer_shape1) {
  ge::op::ResizeBicubic op;
  op.UpdateInputDesc("images", create_desc_with_ori({-2}, ge::DT_INT32, ge::FORMAT_NHWC, {-2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("size", create_desc_with_ori({2, 2}, ge::DT_INT32, ge::FORMAT_NHWC, {2, 2}, ge::FORMAT_NHWC));
  op.UpdateOutputDesc("y", create_desc_with_ori({1, 1, 1,1}, ge::DT_FLOAT, ge::FORMAT_NDHWC,{1, 1, 1,1}, ge::FORMAT_NDHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeBicubic, ResizeBicubic_infer_shape2) {
  ge::op::ResizeBicubic op;
  op.UpdateInputDesc("images", create_desc_with_ori({-2}, ge::DT_INT32, ge::FORMAT_NHWC, {-2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("size", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_NHWC, {2}, ge::FORMAT_NHWC));
  op.UpdateOutputDesc("y", create_desc_with_ori({1, 1, 1,1}, ge::DT_FLOAT, ge::FORMAT_NDHWC,{1, 1, 1,1}, ge::FORMAT_NDHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ResizeBicubic, ResizeBicubic_infer_shape3) {
  ge::op::ResizeBicubic op;
  op.UpdateInputDesc("images", create_desc_with_ori({1, 1, 1, 1}, ge::DT_INT32, ge::FORMAT_NHWC, {1, 1, 1,1}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("size", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_NHWC, {2}, ge::FORMAT_NHWC));
  op.UpdateOutputDesc("y", create_desc_with_ori({1, 1, 1,1}, ge::DT_FLOAT, ge::FORMAT_NDHWC,{1, 1, 1,1}, ge::FORMAT_NDHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ResizeBicubic, ResizeBicubic_infer_shape4) {
  ge::op::ResizeBicubic op;
  op.UpdateInputDesc("images", create_desc_with_ori({-2}, ge::DT_INT32, ge::FORMAT_NCHW, {-2}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("size", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_NCHW, {2}, ge::FORMAT_NCHW));
  op.UpdateOutputDesc("y", create_desc_with_ori({1, 1, 1,1}, ge::DT_FLOAT, ge::FORMAT_NDHWC,{1, 1, 1,1}, ge::FORMAT_NDHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ResizeBicubic, ResizeBicubic_infer_shape5) {
  ge::op::ResizeBicubic op;
  op.UpdateInputDesc("images", create_desc_with_ori({1, 1, 1, 1}, ge::DT_INT32, ge::FORMAT_NCHW, {1, 1, 1,1}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("size", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_NCHW, {2}, ge::FORMAT_NCHW));
  op.UpdateOutputDesc("y", create_desc_with_ori({1, 1, 1,1}, ge::DT_FLOAT, ge::FORMAT_NDHWC,{1, 1, 1,1}, ge::FORMAT_NDHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}