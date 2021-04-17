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
 * @file test_string_format_proto.cpp
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

class ResizeBicubicGrad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ResizeBicubicGrad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ResizeBicubicGrad TearDown" << std::endl;
  }
};

TEST_F(ResizeBicubicGrad, ResizeBicubicGrad_infer_shape) {
  ge::op::ResizeBicubicGrad op;
  op.UpdateInputDesc("grads", create_desc_with_ori({5, 4, 3 ,2}, ge::DT_FLOAT, ge::FORMAT_NHWC, {5, 4, 3 ,2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("original_image", create_desc_with_ori({5, 2, 3 ,2}, ge::DT_FLOAT, ge::FORMAT_NHWC, {5, 2, 3 ,2}, ge::FORMAT_NHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ResizeBicubicGrad, ResizeBicubicGrad_infer_shape1) {
  ge::op::ResizeBicubicGrad op;
  op.UpdateInputDesc("grads", create_desc_with_ori({5, 4, 3 ,2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {5, 4, 3 ,2}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("original_image", create_desc_with_ori({5, 2, 3 ,2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {5, 2, 3 ,2}, ge::FORMAT_NCHW));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}