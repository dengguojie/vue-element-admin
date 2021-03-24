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

class DecodeAndCropJpeg : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DecodeAndCropJpeg SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DecodeAndCropJpeg TearDown" << std::endl;
  }
};

TEST_F(DecodeAndCropJpeg, decode_and_crop_jpeg_infer_shape_success) {
  ge::op::DecodeAndCropJpeg op;

  op.UpdateInputDesc("contents", create_desc({}, ge::DT_STRING));
  op.UpdateInputDesc("crop_window", create_desc({4}, ge::DT_STRING));
  op.SetAttr("channels", 4);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("image");
  std::vector<int64_t> expect_result {-1, -1, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expect_result);
}

TEST_F(DecodeAndCropJpeg, decode_and_crop_jpeg_infer_shape_failed_1) {
  ge::op::DecodeAndCropJpeg op;

  op.UpdateInputDesc("contents", create_desc({}, ge::DT_STRING));
  op.UpdateInputDesc("crop_window", create_desc({4}, ge::DT_STRING));
  op.SetAttr("channels", -1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DecodeAndCropJpeg, decode_and_crop_jpeg_infer_shape_failed_2) {
  ge::op::DecodeAndCropJpeg op;

  op.UpdateInputDesc("contents", create_desc({2}, ge::DT_STRING));
  op.UpdateInputDesc("crop_window", create_desc({4}, ge::DT_STRING));
  op.SetAttr("channels", 4);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DecodeAndCropJpeg, decode_and_crop_jpeg_infer_shape_failed_3) {
  ge::op::DecodeAndCropJpeg op;

  op.UpdateInputDesc("contents", create_desc({}, ge::DT_STRING));
  op.UpdateInputDesc("crop_window", create_desc({4, 2}, ge::DT_STRING));
  op.SetAttr("channels", 4);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DecodeAndCropJpeg, decode_and_crop_jpeg_infer_shape_failed_4) {
  ge::op::DecodeAndCropJpeg op;

  op.UpdateInputDesc("contents", create_desc({}, ge::DT_STRING));
  op.UpdateInputDesc("crop_window", create_desc({2}, ge::DT_STRING));
  op.SetAttr("channels", 4);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}