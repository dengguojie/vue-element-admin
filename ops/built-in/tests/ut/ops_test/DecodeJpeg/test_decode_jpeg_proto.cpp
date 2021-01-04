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

class DecodeJpeg : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DecodeJpeg SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DecodeJpeg TearDown" << std::endl;
  }
};

TEST_F(DecodeJpeg, decode_jpeg_infer_shape_wrong_type) {
  ge::op::DecodeJpeg op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("contents", tensor_desc);
  op.SetAttr("dct_method", "");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}