/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"

class AvgPool1D_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPool1D_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPool1D_UT TearDown" << std::endl;
  }
};

TEST_F(AvgPool1D_UT, VerifyAvgPool1D_001) {
  ge::op::AvgPool1D op;
  op.SetAttr("pads", {});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1D_UT, VerifyAvgPool1D_002) {
  ge::op::AvgPool1D op;
  std::vector<int64_t> pads = {1, 2};
  op.SetAttr("pads", pads);
  op.SetAttr("ksize", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1D_UT, VerifyAvgPool1D_003) {
  ge::op::AvgPool1D op;
  std::vector<int64_t> pads = {1, 2};
  op.SetAttr("pads", pads);
  op.SetAttr("ksize", 1);
  op.SetAttr("strides", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1D_UT, VerifyAvgPool1D_004) {
  ge::op::AvgPool1D op;
  std::vector<int64_t> pads = {1, 2};
  op.SetAttr("pads", pads);
  op.SetAttr("ksize", 1);
  op.SetAttr("strides", 1);
  op.SetAttr("ceil_mode", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1D_UT, VerifyAvgPool1D_005) {
  ge::op::AvgPool1D op;
  std::vector<int64_t> pads = {1, 2};
  op.SetAttr("pads", pads);
  op.SetAttr("ksize", 1);
  op.SetAttr("strides", 1);
  op.SetAttr("count_include_pad", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1D_UT, InfershapeAvgPool1D_001) {
  ge::op::AvgPool1D op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 2, 2, 2, 16}, ge::FORMAT_NCHW));
  op.SetAttr("ksize", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1D_UT, InfershapeAvgPool1D_002) {
  ge::op::AvgPool1D op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 2, 2, 2, 16}, ge::FORMAT_NCHW));
  op.SetAttr("ksize", 1);
  op.SetAttr("strides", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1D_UT, InfershapeAvgPool1D_003) {
  ge::op::AvgPool1D op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 2, 2, 2, 16}, ge::FORMAT_NCHW));
  op.SetAttr("ksize", 1);
  op.SetAttr("strides", 0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1D_UT, InfershapeAvgPool1D_004) {
  ge::op::AvgPool1D op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 2, 2, 2, 16}, ge::FORMAT_NCHW));
  op.SetAttr("ksize", 1);
  op.SetAttr("strides", 1);
  op.SetAttr("pads", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1D_UT, InfershapeAvgPool1D_005) {
  ge::op::AvgPool1D op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 2, 2, 2, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> pads = {1};
  op.SetAttr("ksize", 1);
  op.SetAttr("strides", 1);
  op.SetAttr("pads", pads);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1D_UT, InfershapeAvgPool1D_006) {
  ge::op::AvgPool1D op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 2, 2, 2, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> pads = {1, 2, 3};
  op.SetAttr("ksize", 1);
  op.SetAttr("strides", 1);
  op.SetAttr("pads", pads);
  op.SetAttr("ceil_mode", pads);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1D_UT, InfershapeAvgPool1D_007) {
  ge::op::AvgPool1D op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 2, 2, 16}, ge::FORMAT_ND));
  std::vector<int32_t> pads = {1, 2, 3};
  op.SetAttr("ksize", 1);
  op.SetAttr("strides", 1);
  op.SetAttr("pads", pads);
  op.SetAttr("ceil_mode", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1D_UT, InfershapeAvgPool1D_008) {
  ge::op::AvgPool1D op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 2, 2, 2, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> pads = {1, 2, 3};
  op.SetAttr("ksize", 1);
  op.SetAttr("strides", 1);
  op.SetAttr("pads", pads);
  op.SetAttr("ceil_mode", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 2, 2, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}