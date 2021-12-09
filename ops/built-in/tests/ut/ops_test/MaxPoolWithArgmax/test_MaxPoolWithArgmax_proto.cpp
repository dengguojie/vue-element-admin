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

class MaxPoolWithArgmax_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaxPoolWithArgmax_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPoolWithArgmax_UT TearDown" << std::endl;
  }
};

TEST_F(MaxPoolWithArgmax_UT, VerifyMaxPoolWithArgmax_001) {
  ge::op::MaxPoolWithArgmax op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolWithArgmax_UT, VerifyMaxPoolWithArgmax_002) {
  ge::op::MaxPoolWithArgmax op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {5, 5, 5};
  op.SetAttr("ksize", ksizeList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolWithArgmax_UT, VerifyMaxPoolWithArgmax_003) {
  ge::op::MaxPoolWithArgmax op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolWithArgmax_UT, VerifyMaxPoolWithArgmax_004) {
  ge::op::MaxPoolWithArgmax op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolWithArgmax_UT, VerifyMaxPoolWithArgmax_005) {
  ge::op::MaxPoolWithArgmax op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 5};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolWithArgmax_UT, VerifyMaxPoolWithArgmax_006) {
  ge::op::MaxPoolWithArgmax op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {1, 255, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolWithArgmax_UT, VerifyMaxPoolWithArgmax_007) {
  ge::op::MaxPoolWithArgmax op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", ksizeList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolWithArgmax_UT, VerifyMaxPoolWithArgmax_008) {
  ge::op::MaxPoolWithArgmax op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "error");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolWithArgmax_UT, VerifyMaxPoolWithArgmax_009) {
  ge::op::MaxPoolWithArgmax op;
  auto tensor_desc = create_desc_with_ori({2, 2, 2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2, 2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolWithArgmax_UT, VerifyMaxPoolWithArgmax_010) {
  ge::op::MaxPoolWithArgmax op;
  auto tensor_desc = create_desc_with_ori({5, 5, 5, 5}, ge::DT_INT64, ge::FORMAT_NCHW, {5, 5, 5, 5}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 2, 2, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_output_shape_y = {5, 5, 2, 2};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape_y);

  auto output_argmax_desc = op.GetOutputDesc("argmax");
  EXPECT_EQ(output_argmax_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_output_shape_argmax = {5, 5, 2, 2};
  EXPECT_EQ(output_argmax_desc.GetShape().GetDims(), expected_output_shape_argmax);
}