/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_MaxPoolGradWithArgmaxV2_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"

class MaxPoolWithArgmaxV2Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaxPoolWithArgmaxV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPoolWithArgmaxV2 TearDown" << std::endl;
  }
};

TEST_F(MaxPoolWithArgmaxV2Test, max_pool_with_argmax_v2_0) {
  ge::op::MaxPoolWithArgmaxV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {56, 56}, {56, 56}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 4, 56, 56, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("pads", {1, 1, 1, 1});
  op.SetAttr("ceil_mode", false);
  std::vector<int64_t> expect_output_shape = {-1, 4, 28, 28, 16};
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y1_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expect_output_shape);
}

TEST_F(MaxPoolWithArgmaxV2Test, max_pool_with_argmax_v2_1) {
  ge::op::MaxPoolWithArgmaxV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {2, 2}, {1, -1}, {10, 10}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 2, -1, 10, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 32, 10, 10},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("pads", {1, 1, 1, 1});
  op.SetAttr("ceil_mode", false);
  std::vector<int64_t> expect_output_shape = {-1, 2, -1, 10, 16};
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y1_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expect_output_shape);
}

TEST_F(MaxPoolWithArgmaxV2Test, max_pool_with_argmax_v2_2) {
  ge::op::MaxPoolWithArgmaxV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {2, 2}, {8, 12}, {10, 10}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 2, -1, 10, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 32, 10, 10},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("pads", {1, 1, 1, 1});
  op.SetAttr("ceil_mode", false);
  std::vector<int64_t> expect_output_shape = {-1, 2, -1, 10, 16};
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y1_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expect_output_shape);
}