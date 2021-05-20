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

class AudioSpectrogram : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AudioSpectrogram SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AudioSpectrogram TearDown" << std::endl;
  }
};

TEST_F(AudioSpectrogram, AudioSpectrogram_infer_shape) {
  ge::op::AudioSpectrogram op;
  op.UpdateInputDesc("x", create_desc_with_ori({4, 3 ,2}, ge::DT_FLOAT, ge::FORMAT_NHWC, { 4, 3 ,2}, ge::FORMAT_NHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}


TEST_F(AudioSpectrogram, AudioSpectrogram_infer_shape3) {
  ge::op::AudioSpectrogram op;
  op.UpdateInputDesc("x", create_desc_with_ori({4, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC, { 4, 2 }, ge::FORMAT_NHWC));
  op.SetAttr("stride", 0);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}