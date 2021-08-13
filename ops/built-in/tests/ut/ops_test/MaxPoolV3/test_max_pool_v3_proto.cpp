/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the
 License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_MaxPoolV3_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"

using namespace ge;
using namespace op;

class MaxPoolV3Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaxPoolV3 Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPoolV3 Proto Test TearDown" << std::endl;
  }
};

TEST_F(MaxPoolV3Test, max_pool_v3_infershape_diff_test) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {56, 56}, {56, 56}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 4, 56, 56, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", {1, 1, 1, 1});
  op.SetAttr("data_format", "NC1HWC0");
  op.SetAttr("global_pooling", false);
  op.SetAttr("ceil_mode", false);
  std::vector<int64_t> expect_output_shape = {-1, 4, 28, 28, 16};
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y1_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expect_output_shape);
}
TEST_F(MaxPoolV3Test, max_pool_v3_global_true) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {3, 3}, {3, 3}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 4, 3, 3, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 3, 3},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", {1, 1, 1, 1});
  op.SetAttr("data_format", "NC1HWC0");
  op.SetAttr("global_pooling", true);
  op.SetAttr("ceil_mode", false);
  std::vector<int64_t> expect_output_shape = {-1, 4, 1, 1, 16};
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y1_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expect_output_shape);
}
TEST_F(MaxPoolV3Test, max_pool_v3_infershape_ceilmode_true) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {56, 56}, {56, 56}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 4, 56, 56, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", {1, 1, 1, 1});
  op.SetAttr("data_format", "NC1HWC0");
  op.SetAttr("global_pooling", false);
  op.SetAttr("ceil_mode", true);
  std::vector<int64_t> expect_output_shape = {-1, 4, 29, 29, 16};
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y1_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expect_output_shape);
}
TEST_F(MaxPoolV3Test, max_pool_v3_infershape_failed) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {64, 64}, {56, 56}, {56, 56}};
  auto tensor_desc = create_desc_shape_range({-1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 64, 56, 56},
                                             ge::FORMAT_NCHW, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", {1, 1, 1, 1});
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("global_pooling", false);
  op.SetAttr("ceil_mode", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(MaxPoolV3Test, max_pool_v3_infershape_same) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {56, 56}, {56, 56}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 4, 56, 56, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("pads", {0, 0, 0, 0});
  op.SetAttr("data_format", "NC1HWC0");
  op.SetAttr("global_pooling", false);
  op.SetAttr("ceil_mode", false);
  std::vector<int64_t> expect_output_shape = {-1, 4, 28, 28, 16};
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y1_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expect_output_shape);
}
TEST_F(MaxPoolV3Test, max_pool_v3_infershape_withoutksize) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {56, 56}, {56, 56}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 4, 56, 56, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("pads", {0, 0, 0, 0});
  op.SetAttr("data_format", "NC1HWC0");
  op.SetAttr("global_pooling", false);
  op.SetAttr("ceil_mode", false);
  std::vector<int64_t> expect_output_shape = {-1, 4, 28, 28, 16};
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(MaxPoolV3Test, max_pool_v3_infershape_withoutstrides) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {56, 56}, {56, 56}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 4, 56, 56, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("pads", {0, 0, 0, 0});
  op.SetAttr("data_format", "NC1HWC0");
  op.SetAttr("global_pooling", false);
  op.SetAttr("ceil_mode", false);
  std::vector<int64_t> expect_output_shape = {-1, 4, 28, 28, 16};
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(MaxPoolV3Test, max_pool_v3_infershape_withoutdataformat) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {56, 56}, {56, 56}, {64, 64}};
  auto tensor_desc = create_desc_shape_range({-1, 56, 56, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 56, 56, 64},
                                             ge::FORMAT_NHWC, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", {1, 1, 1, 1});
  op.SetAttr("global_pooling", false);
  op.SetAttr("ceil_mode", false);
  std::vector<int64_t> expect_output_shape = {-1, 4, 28, 28, 16};
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(MaxPoolV3Test, max_pool_v3_infershape_dynamicdim) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {55, 57}, {56, 56}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 4, -1, 56, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", {1, 1, 1, 1});
  op.SetAttr("data_format", "NC1HWC0");
  op.SetAttr("global_pooling", false);
  op.SetAttr("ceil_mode", false);
  std::vector<int64_t> expect_output_shape = {-1, 4, -1, 28, 16};
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y1_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expect_output_shape);
}
TEST_F(MaxPoolV3Test, max_pool_v3_infershape_stridesfailed) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {56, 56}, {56, 56}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 4, 56, 56, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 0, 0});
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", {1, 1, 1, 1});
  op.SetAttr("data_format", "NC1HWC0");
  op.SetAttr("global_pooling", false);
  op.SetAttr("ceil_mode", false);
  std::vector<int64_t> expect_output_shape = {-1, 4, 28, 28, 16};
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}