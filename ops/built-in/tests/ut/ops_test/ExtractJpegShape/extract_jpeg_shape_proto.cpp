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

class ExtractJpegShape : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ExtractJpegShape SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ExtractJpegShape TearDown" << std::endl;
  }
};

TEST_F(ExtractJpegShape, ExtractJpegShape_infer_shape) {
  ge::op::ExtractJpegShape op;
  op.UpdateInputDesc("contents", create_desc({1}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ExtractJpegShape, ExtractJpegShape_infer_shape2) {
  ge::op::ExtractJpegShape op;
  op.UpdateInputDesc("contents", create_desc({}, ge::DT_FLOAT));
  op.SetAttr("N",1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}