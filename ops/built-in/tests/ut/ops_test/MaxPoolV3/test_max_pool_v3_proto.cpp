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
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"

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
  auto output_y1_desc = op.GetOutputDescByName("y");
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
  auto output_y1_desc = op.GetOutputDescByName("y");
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
  auto output_y1_desc = op.GetOutputDescByName("y");
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
  auto output_y1_desc = op.GetOutputDescByName("y");
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
  auto output_y1_desc = op.GetOutputDescByName("y");
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

TEST_F(MaxPoolV3Test, InfershapeMaxPoolV3_001) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {55, 57}, {56, 56}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 4, -1, 56, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("data_format", {});

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, InfershapeMaxPoolV3_002) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {55, 57}, {56, 56}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 4, -1, 56, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("data_format", "NC1HWC0");
  op.SetAttr("padding_mode", {});

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, InfershapeMaxPoolV3_003) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {55, 57}, {56, 56}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 4, -1, 56, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("data_format", "NC1HWC0");
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, InfershapeMaxPoolV3_004) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {55, 57}, {56, 56}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 4, -1, 56, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("data_format", "NC1HWC0");
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", {1, 1, 1});

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, InfershapeMaxPoolV3_005) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {55, 57}, {56, 56}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 4, -1, 56, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("data_format", "NC1HWC0");
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", {1, 1, 1, 1});
  op.SetAttr("global_pooling", "false");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, InfershapeMaxPoolV3_006) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {55, 57}, {56, 56}, {16, 16}};
  auto tensor_desc = create_desc_shape_range({-1, 4, -1, 56, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("data_format", "NC1HWC0");
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", {1, 1, 1, 1});
  op.SetAttr("global_pooling", true);
  op.SetAttr("ceil_mode", "true");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, InfershapeMaxPoolV3_007) {
  ge::op::MaxPoolV3 op;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges = {{1, 6}, {4, 4}, {55, 57}, {56, 56}};
  auto tensor_desc = create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56},
                                             ge::FORMAT_NC1HWC0, shape_ranges);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", {1, 1, 3, 3});
  op.SetAttr("strides", {1, 1, 2, 2});
  op.SetAttr("data_format", "NC1HWC0");
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", {1, 1, 1, 1});
  op.SetAttr("global_pooling", true);
  op.SetAttr("ceil_mode", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_001) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56}, ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_002) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56}, ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 2};
  op.SetAttr("ksize", ksize);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_003) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56}, ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 2, 3, 4};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_004) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56}, ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 2, 3, 4};
  std::vector<int64_t> strides = {1, 2, 3};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_005) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56}, ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 2, 3, 4};
  std::vector<int64_t> strides = {1, 2, 3, 4};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", strides);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_006) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56}, ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 2, 3, 4};
  std::vector<int64_t> strides = {1, 2, 3, 4};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "error");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_007) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56}, ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 2, 3, 4};
  std::vector<int64_t> strides = {1, 2, 3, 4};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_008) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56}, ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 2, 3, 4};
  std::vector<int64_t> strides = {1, 2, 3, 4};
  std::vector<int64_t> pads = {1, 2};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_009) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56}, ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 2, 3, 4};
  std::vector<int64_t> strides = {1, 2, 3, 4};
  std::vector<int64_t> pads = {1, 2, 3, 4};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", pads);
  op.SetAttr("data_format", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_010) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56}, ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 2, 3, 4};
  std::vector<int64_t> strides = {1, 2, 3, 4};
  std::vector<int64_t> pads = {1, 2, 3, 4};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", pads);
  op.SetAttr("data_format", "NCHW");
  op.SetAttr("ceil_mode", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_011) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56}, ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 2, 3, 4};
  std::vector<int64_t> strides = {1, 2, 3, 4};
  std::vector<int64_t> pads = {1, 2, 3, 4};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", pads);
  op.SetAttr("data_format", "ND");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_012) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56}, ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {2, 2, 3, 4};
  std::vector<int64_t> strides = {2, 2, 3, 4};
  std::vector<int64_t> pads = {1, 2, 3, 4};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", pads);
  op.SetAttr("data_format", "NCHW");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_013) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56}, ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 1, 3, 4};
  std::vector<int64_t> strides = {1, 1, 3, 4};
  std::vector<int64_t> pads = {8, 8, 8, 8};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", pads);
  op.SetAttr("data_format", "NCHW");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_014) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56}, ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 1, 3, 4};
  std::vector<int64_t> strides = {1, 1, 3, 4};
  std::vector<int64_t> pads = {8, 8, 8, 8};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", pads);
  op.SetAttr("data_format", "NHWC");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_015) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1, 64, 56, 56}, ge::FORMAT_NC1HWC0);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 1, 3, 1};
  std::vector<int64_t> strides = {1, 1, 3, 1};
  std::vector<int64_t> pads = {8, 8, 8, 8};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", pads);
  op.SetAttr("data_format", "NHWC");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_016) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 64, 56, 56}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 3, 3, 1};
  std::vector<int64_t> strides = {1, 3, 3, 1};
  std::vector<int64_t> pads = {1, 1, 1, 1};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", pads);
  op.SetAttr("data_format", "NHWC");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, VerifyMaxPoolV3_017) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 64, 56, 56}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 3, 3, 1};
  std::vector<int64_t> strides = {1, 3, 3, 1};
  std::vector<int64_t> pads = {1, 1, 1, 1};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "VALID");
  op.SetAttr("pads", pads);
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("ceil_mode", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Test, InferDataSlice1) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 64, 56, 56}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 3, 3, 1};
  std::vector<int64_t> strides = {1, 3, 3, 1};
  std::vector<int64_t> pads = {1, 1, 1, 1};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "VALID");
  op.SetAttr("pads", pads);
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("ceil_mode", true);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_NE(status, ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolV3Test, InferDataSlice2) {
  ge::op::MaxPoolV3 op;
  auto tensor_desc =
      create_desc_with_ori({1, 64, 56, 56}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 64, 56, 56}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int64_t> ksize = {1, 3, 3, 1};
  std::vector<int64_t> strides = {1, 3, 3, 1};
  std::vector<int64_t> pads = {1, 1, 1, 1};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "VALID");
  op.SetAttr("pads", pads);
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("ceil_mode", true);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  std::vector<std::vector<int64_t>> y_data_slice = {{0, 2}, {}, {}, {}};
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto x_desc = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(x_desc, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  EXPECT_EQ(x_data_slice, y_data_slice);
}
