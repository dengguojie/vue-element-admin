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

class Mfcc : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Mfcc SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Mfcc TearDown" << std::endl;
  }
};

TEST_F(Mfcc, Mfcc_infer_shape1) {
  ge::op::Mfcc op;
  op.UpdateInputDesc("spectrogram", create_desc_with_ori({4, 3 ,2}, ge::DT_FLOAT, ge::FORMAT_NHWC, { 4, 3 ,2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("sample_rate", create_desc_with_ori({2,3,4}, ge::DT_INT32, ge::FORMAT_NHWC, {2,3,4}, ge::FORMAT_NHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Mfcc, Mfcc_infer_shape2) {
  ge::op::Mfcc op;
  op.UpdateInputDesc("spectrogram", create_desc_with_ori({4, 3 ,2}, ge::DT_FLOAT, ge::FORMAT_NHWC, { 4, 3 ,2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("sample_rate", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_NHWC, {2}, ge::FORMAT_NHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Mfcc, Mfcc_infer_shape3) {
  ge::op::Mfcc op;
  op.UpdateInputDesc("spectrogram", create_desc_with_ori({4, 3 ,2}, ge::DT_FLOAT, ge::FORMAT_NHWC, { 4, 3 ,2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("sample_rate", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_NHWC, {2}, ge::FORMAT_NHWC));
  op.SetAttr("dct_coefficient_count",-1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Mfcc, Mfcc_infer_shape4) {
  ge::op::Mfcc op;
  op.UpdateInputDesc("spectrogram", create_desc_with_ori({4, 3 ,2}, ge::DT_FLOAT, ge::FORMAT_NHWC, { 4, 3 ,2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("sample_rate", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_NHWC, {2}, ge::FORMAT_NHWC));
  op.SetAttr("filterbank_channel_count",20);
  op.SetAttr("dct_coefficient_count",30);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Mfcc, Mfcc_infer_shape5) {
  ge::op::Mfcc op;
  op.UpdateInputDesc("spectrogram", create_desc_with_ori({4, 3}, ge::DT_FLOAT, ge::FORMAT_NHWC, { 4, 3}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("sample_rate", create_desc_with_ori({0}, ge::DT_INT32, ge::FORMAT_NHWC, {2}, ge::FORMAT_NHWC));
  op.SetAttr("filterbank_channel_count",20);
  op.SetAttr("dct_coefficient_count",30);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Mfcc, Mfcc_infer_shape6) {
  ge::op::Mfcc op;
  op.UpdateInputDesc("spectrogram", create_desc_with_ori({4, 3, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC, { 4, 3, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("sample_rate", create_desc_with_ori({0}, ge::DT_INT64, ge::FORMAT_NHWC, {2}, ge::FORMAT_NHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Mfcc, Mfcc_infer_shape7) {
  ge::op::Mfcc op;
  op.UpdateInputDesc("spectrogram", create_desc_with_ori({4, 3, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC, { 4, 3, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("sample_rate", create_desc_with_ori({0}, ge::DT_INT32, ge::FORMAT_NHWC, {2}, ge::FORMAT_NHWC));
  op.SetAttr("dct_coefficient_count", -2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}


TEST_F(Mfcc, Mfcc_infer_shape8) {
  ge::op::Mfcc op;
  op.UpdateInputDesc("spectrogram", create_desc_with_ori({4, 3 ,2}, ge::DT_FLOAT, ge::FORMAT_NHWC, { 4, 3 ,2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("sample_rate", create_desc_with_ori({0}, ge::DT_INT32, ge::FORMAT_NHWC, {0}, ge::FORMAT_NHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}