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

class MaxPoolGradGradWithArgmax_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaxPoolGradGradWithArgmax_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPoolGradGradWithArgmax_UT TearDown" << std::endl;
  }
};

TEST_F(MaxPoolGradGradWithArgmax_UT, VerifyMaxPoolGradGradWithArgmax_001) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc("x", create_desc({2, 2}, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({2, 2}, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradGradWithArgmax_UT, InfershapeMaxPoolGradGradWithArgmax_001) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc("x", create_desc_with_ori({2, 2}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW));
  op.SetAttr("ksize", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradGradWithArgmax_UT, InfershapeMaxPoolGradGradWithArgmax_002) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc("x", create_desc_with_ori({2, 2}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {5, 5, 5};
  op.SetAttr("ksize", ksizeList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradGradWithArgmax_UT, InfershapeMaxPoolGradGradWithArgmax_003) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc("x", create_desc_with_ori({2, 2}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradGradWithArgmax_UT, InfershapeMaxPoolGradGradWithArgmax_004) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc("x", create_desc_with_ori({2, 2}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradGradWithArgmax_UT, InfershapeMaxPoolGradGradWithArgmax_005) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc("x", create_desc_with_ori({2, 2}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 0, 5};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradGradWithArgmax_UT, InfershapeMaxPoolGradGradWithArgmax_006) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc("x", create_desc_with_ori({2, 2}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", ksizeList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradGradWithArgmax_UT, InfershapeMaxPoolGradGradWithArgmax_007) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc("x", create_desc_with_ori({2, 2}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "error");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradGradWithArgmax_UT, InfershapeMaxPoolGradGradWithArgmax_008) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {6, 6, 6, 16}, ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "grad", create_desc_with_ori({2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {2, 2, 2, 16}, ge::FORMAT_NHWC));
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");
  op.SetAttr("data_format", "NHWC");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_y = {6, 2, 2, 16};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape_y);
}

TEST_F(MaxPoolGradGradWithArgmax_UT, InfershapeMaxPoolGradGradWithArgmax_009) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {6, 6, 6, 16}, ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "grad", create_desc_with_ori({2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {2, 2, 2, 16}, ge::FORMAT_NHWC));
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");
  op.SetAttr("data_format", "NCHW");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_y = {6, 2, 2, 16};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape_y);
}

TEST_F(MaxPoolGradGradWithArgmax_UT, InfershapeMaxPoolGradGradWithArgmax_010) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {6, 6, 6, 16}, ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "grad", create_desc_with_ori({2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {2, 2, 2, 16}, ge::FORMAT_NHWC));
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("data_format", "NHWC");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_y = {6, 1, 1, 16};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape_y);
}

TEST_F(MaxPoolGradGradWithArgmax_UT, InfershapeMaxPoolGradGradWithArgmax_011) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {6, 6, 6, 16}, ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "grad", create_desc_with_ori({2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {2, 2, 2, 16}, ge::FORMAT_NHWC));
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("data_format", "NCHW");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_y = {6, 1, 1, 16};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape_y);
}

TEST_F(MaxPoolGradGradWithArgmax_UT, InfershapeMaxPoolGradGradWithArgmax_012) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6, 6, 6, 16}, ge::FORMAT_NCHW));
  op.UpdateInputDesc(
      "grad", create_desc_with_ori({2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 2, 2, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");
  op.SetAttr("data_format", "NHWC");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_y = {6, 6, 2, 6};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape_y);
}

TEST_F(MaxPoolGradGradWithArgmax_UT, InfershapeMaxPoolGradGradWithArgmax_013) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6, 6, 6, 16}, ge::FORMAT_NCHW));
  op.UpdateInputDesc(
      "grad", create_desc_with_ori({2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 2, 2, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");
  op.SetAttr("data_format", "NCHW");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_y = {6, 6, 2, 6};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape_y);
}

TEST_F(MaxPoolGradGradWithArgmax_UT, InfershapeMaxPoolGradGradWithArgmax_014) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6, 6, 6, 16}, ge::FORMAT_NCHW));
  op.UpdateInputDesc(
      "grad", create_desc_with_ori({2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 2, 2, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("data_format", "NHWC");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_y = {6, 6, 1, 4};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape_y);
}

TEST_F(MaxPoolGradGradWithArgmax_UT, InfershapeMaxPoolGradGradWithArgmax_015) {
  ge::op::MaxPoolGradGradWithArgmax op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6, 6, 6, 16}, ge::FORMAT_NCHW));
  op.UpdateInputDesc(
      "grad", create_desc_with_ori({2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 2, 2, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("data_format", "NCHW");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_y = {6, 6, 1, 4};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape_y);
}