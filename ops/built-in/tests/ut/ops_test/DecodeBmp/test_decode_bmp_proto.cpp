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
 * @file test_decode_jpeg_proto.cpp
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

class DecodeBmpTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DecodeBmpTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DecodeBmpTest TearDown" << std::endl;
  }
};

TEST_F(DecodeBmpTest, infershape_success) {
  ge::op::DecodeBmp op;
  op.UpdateInputDesc("contents", create_desc({}, ge::DT_STRING));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DecodeBmpTest, infershape_failed_1) {
  ge::op::DecodeBmp op;
  op.UpdateInputDesc("contents", create_desc({}, ge::DT_STRING));
  op.SetAttr("channels", 5);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DecodeBmpTest, infershape_failed_2) {
  ge::op::DecodeBmp op;
  op.UpdateInputDesc("contents", create_desc({}, ge::DT_FLOAT));
  op.SetAttr("channels", 5);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}