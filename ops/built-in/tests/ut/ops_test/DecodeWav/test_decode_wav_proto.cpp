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
#include "audio_ops.h"

class DecodeWav : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DecodeWav SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DecodeWav TearDown" << std::endl;
  }
};

TEST_F(DecodeWav, DecodeWav_infer_shape) {
  ge::op::DecodeWav op;
  op.UpdateInputDesc("contents", create_desc({1}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DecodeWav, DecodeWav_infer_shape2) {
  ge::op::DecodeWav op;
  op.UpdateInputDesc("contents", create_desc({}, ge::DT_STRING));
  op.SetAttr("desired_channels", -2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DecodeWav, DecodeWav_infer_shape3) {
  ge::op::DecodeWav op;
  op.UpdateInputDesc("contents", create_desc({}, ge::DT_STRING));
  op.SetAttr("desired_channels", -1);
  op.SetAttr("desired_samples", -2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DecodeWav, DecodeWav_infer_shape5) {
  ge::op::DecodeWav op;
  op.UpdateInputDesc("contents", create_desc({}, ge::DT_STRING));
  op.SetAttr("desired_channel", -1);
  op.SetAttr("desired_samples", -2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
