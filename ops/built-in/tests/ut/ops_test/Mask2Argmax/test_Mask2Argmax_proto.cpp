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

class Mask2Argmax_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Mask2Argmax_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Mask2Argmax_UT TearDown" << std::endl;
  }
};

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_001) {
  ge::op::Mask2Argmax op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("ksize", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_002) {
  ge::op::Mask2Argmax op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {5, 5, 5};
  op.SetAttr("ksize", ksizeList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_003) {
  ge::op::Mask2Argmax op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_004) {
  ge::op::Mask2Argmax op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_005) {
  ge::op::Mask2Argmax op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 5};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_006) {
  ge::op::Mask2Argmax op;
  auto tensor_desc = create_desc_with_ori({2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2, 2}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {1, 255, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_007) {
  ge::op::Mask2Argmax op;
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

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_008) {
  ge::op::Mask2Argmax op;
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

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_009) {
  ge::op::Mask2Argmax op;
  auto tensor_desc = create_desc_with_ori({5, 5, 5, 5}, ge::DT_INT64, ge::FORMAT_NHWC, {5, 5, 5, 5}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 2, 2, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_argmax_desc = op.GetOutputDesc("argmax");
  EXPECT_EQ(output_argmax_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape_argmax = {5, 3, 3, 5};
  EXPECT_EQ(output_argmax_desc.GetShape().GetDims(), expected_output_shape_argmax);
}

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_010) {
  ge::op::Mask2Argmax op;
  auto tensor_desc = create_desc_with_ori({5, 5, 5, 5}, ge::DT_INT64, ge::FORMAT_NHWC, {5, 5, 5, 5}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 2, 2, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_argmax_desc = op.GetOutputDesc("argmax");
  EXPECT_EQ(output_argmax_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape_argmax = {5, 2, 2, 5};
  EXPECT_EQ(output_argmax_desc.GetShape().GetDims(), expected_output_shape_argmax);
}

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_011) {
  ge::op::Mask2Argmax op;
  auto tensor_desc = create_desc_with_ori({5, 5, 5, 5}, ge::DT_INT64, ge::FORMAT_NCHW, {5, 5, 5, 5}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 2, 2, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_argmax_desc = op.GetOutputDesc("argmax");
  EXPECT_EQ(output_argmax_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape_argmax = {5, 5, 3, 3};
  EXPECT_EQ(output_argmax_desc.GetShape().GetDims(), expected_output_shape_argmax);
}

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_012) {
  ge::op::Mask2Argmax op;
  auto tensor_desc = create_desc_with_ori({5, 5, 5, 5}, ge::DT_INT64, ge::FORMAT_NCHW, {5, 5, 5, 5}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 2, 2, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_argmax_desc = op.GetOutputDesc("argmax");
  EXPECT_EQ(output_argmax_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape_argmax = {5, 5, 2, 2};
  EXPECT_EQ(output_argmax_desc.GetShape().GetDims(), expected_output_shape_argmax);
}

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_013) {
  ge::op::Mask2Argmax op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}, {133, 133}, {5, 23}, {13, 25}};
  auto tensor_desc = create_desc_shape_range({-1, -1, 150, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                             {-1, -1, 150, 16}, ge::FORMAT_NCHW, range_x1);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 2, 2, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_argmax_desc = op.GetOutputDesc("argmax");
  EXPECT_EQ(output_argmax_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape_argmax = {-1, -1, 75, 8};
  EXPECT_EQ(output_argmax_desc.GetShape().GetDims(), expected_output_shape_argmax);
}

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_014) {
  ge::op::Mask2Argmax op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}, {133, 133}, {5, 23}, {13, 25}};
  auto tensor_desc = create_desc_shape_range({-1, -1, 150, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                             {-1, -1, 150, 16}, ge::FORMAT_NCHW, range_x1);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 2, 2, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_argmax_desc = op.GetOutputDesc("argmax");
  EXPECT_EQ(output_argmax_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape_argmax = {-1, -1, 75, 8};
  EXPECT_EQ(output_argmax_desc.GetShape().GetDims(), expected_output_shape_argmax);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{30, 30}, {133, 133}, {75, 75}, {8, 8}};
  EXPECT_EQ(output_argmax_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);

}

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_015) {
  ge::op::Mask2Argmax op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}, {133, 133}, {5, 23}, {13, 25}};
  auto tensor_desc = create_desc_shape_range({-1, -1, 150, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                             {-1, -1, 150, 16}, ge::FORMAT_NCHW, range_x1);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 2, 2, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_argmax_desc = op.GetOutputDesc("argmax");
  EXPECT_EQ(output_argmax_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape_argmax = {-1, -1, 75, 8};
  EXPECT_EQ(output_argmax_desc.GetShape().GetDims(), expected_output_shape_argmax);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{30, 30}, {133, 133}, {75, 75}, {8, 8}};
  EXPECT_EQ(output_argmax_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(Mask2Argmax_UT, InfershapeMask2Argmax_016) {
  ge::op::Mask2Argmax op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}, {133, 133}, {5, 23}, {13, 25}};
  auto tensor_desc = create_desc_shape_range({1, 1, 150, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                             {1, 1, 150, 16}, ge::FORMAT_NCHW, range_x1);
  op.UpdateInputDesc("x", tensor_desc);
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 2, 2, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_argmax_desc = op.GetOutputDesc("argmax");
  EXPECT_EQ(output_argmax_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape_argmax = {1, 1, 75, 8};
  EXPECT_EQ(output_argmax_desc.GetShape().GetDims(), expected_output_shape_argmax);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {};
  EXPECT_EQ(output_argmax_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}
