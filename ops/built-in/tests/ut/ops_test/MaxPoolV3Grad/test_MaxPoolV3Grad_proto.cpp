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

class MaxPoolV3Grad_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaxPoolV3Grad_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPoolV3Grad_UT TearDown" << std::endl;
  }
};

TEST_F(MaxPoolV3Grad_UT, VerifyMaxPoolV3Grad_001) {
  ge::op::MaxPoolV3Grad op;
  op.UpdateInputDesc("orig_input", create_desc({2, 2}, ge::DT_FLOAT16));
  op.UpdateInputDesc("orig_output", create_desc({2, 2}, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({2, 2}, ge::DT_FLOAT));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Grad_UT, VerifyMaxPoolV3Grad_002) {
  ge::op::MaxPoolV3Grad op;
  auto input_desc = create_desc({2, 2}, ge::DT_FLOAT16);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);
  op.UpdateInputDesc("grad", input_desc);
  op.SetAttr("data_format", "ND");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Grad_UT, VerifyMaxPoolV3Grad_003) {
  ge::op::MaxPoolV3Grad op;
  auto input_desc = create_desc({2, 2}, ge::DT_FLOAT16);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);
  op.UpdateInputDesc("grad", input_desc);
  op.SetAttr("ksize", {});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Grad_UT, VerifyMaxPoolV3Grad_004) {
  ge::op::MaxPoolV3Grad op;
  auto input_desc = create_desc({2, 2}, ge::DT_FLOAT16);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);
  op.UpdateInputDesc("grad", input_desc);
  std::vector<int64_t> ksize = {1, 2, 3};
  op.SetAttr("ksize", ksize);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Grad_UT, VerifyMaxPoolV3Grad_005) {
  ge::op::MaxPoolV3Grad op;
  auto input_desc = create_desc({2, 2}, ge::DT_FLOAT16);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);
  op.UpdateInputDesc("grad", input_desc);
  std::vector<int64_t> ksize = {2, 2, 3, 4};
  op.SetAttr("data_format", "NCHW");
  op.SetAttr("ksize", ksize);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Grad_UT, VerifyMaxPoolV3Grad_006) {
  ge::op::MaxPoolV3Grad op;
  auto input_desc = create_desc({2, 2}, ge::DT_FLOAT16);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);
  op.UpdateInputDesc("grad", input_desc);
  std::vector<int64_t> ksize = {2, 2, 3, 4};
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("ksize", ksize);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Grad_UT, VerifyMaxPoolV3Grad_007) {
  ge::op::MaxPoolV3Grad op;
  auto input_desc = create_desc({2, 2}, ge::DT_FLOAT16);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);
  op.UpdateInputDesc("grad", input_desc);
  std::vector<int64_t> ksize = {1, 2, 3, 1};
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", {});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Grad_UT, VerifyMaxPoolV3Grad_008) {
  ge::op::MaxPoolV3Grad op;
  auto input_desc = create_desc({2, 2}, ge::DT_FLOAT16);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);
  op.UpdateInputDesc("grad", input_desc);
  std::vector<int64_t> ksize = {1, 2, 3, 1};
  std::vector<int64_t> strides = {1, 2, 3};
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
TEST_F(MaxPoolV3Grad_UT, VerifyMaxPoolV3Grad_009) {
  ge::op::MaxPoolV3Grad op;
  auto input_desc = create_desc({2, 2}, ge::DT_FLOAT16);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);
  op.UpdateInputDesc("grad", input_desc);
  std::vector<int64_t> ksize = {1, 2, 3, 1};
  std::vector<int64_t> strides = {2, 2, 3, 2};
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Grad_UT, VerifyMaxPoolV3Grad_010) {
  ge::op::MaxPoolV3Grad op;
  auto input_desc = create_desc({2, 2}, ge::DT_FLOAT16);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);
  op.UpdateInputDesc("grad", input_desc);
  std::vector<int64_t> ksize = {1, 1, 3, 1};
  std::vector<int64_t> strides = {2, 2, 3, 2};
  op.SetAttr("data_format", "NCHW");
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Grad_UT, VerifyMaxPoolV3Grad_011) {
  ge::op::MaxPoolV3Grad op;
  auto input_desc = create_desc({2, 2}, ge::DT_FLOAT16);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);
  op.UpdateInputDesc("grad", input_desc);
  std::vector<int64_t> ksize = {1, 1, 1, 1};
  std::vector<int64_t> strides = {1, 1, 3, 2};
  op.SetAttr("data_format", "NCHW");
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", strides);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Grad_UT, VerifyMaxPoolV3Grad_012) {
  ge::op::MaxPoolV3Grad op;
  auto input_desc = create_desc({2, 2}, ge::DT_FLOAT16);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);
  op.UpdateInputDesc("grad", input_desc);
  std::vector<int64_t> ksize = {1, 1, 1, 1};
  std::vector<int64_t> strides = {1, 1, 3, 2};
  op.SetAttr("data_format", "NCHW");
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "error");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Grad_UT, VerifyMaxPoolV3Grad_013) {
  ge::op::MaxPoolV3Grad op;
  auto input_desc = create_desc({2, 2}, ge::DT_FLOAT16);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);
  op.UpdateInputDesc("grad", input_desc);
  std::vector<int64_t> ksize = {1, 1, 1, 1};
  std::vector<int64_t> strides = {1, 1, 3, 2};
  op.SetAttr("data_format", "NCHW");
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("pads", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Grad_UT, VerifyMaxPoolV3Grad_014) {
  ge::op::MaxPoolV3Grad op;
  auto input_desc = create_desc({2, 2}, ge::DT_FLOAT16);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);
  op.UpdateInputDesc("grad", input_desc);
  std::vector<int64_t> ksize = {1, 1, 1, 1};
  std::vector<int64_t> strides = {1, 1, 3, 2};
  std::vector<int64_t> pads = {1, 1, 3};
  op.SetAttr("data_format", "NCHW");
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("pads", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3Grad_UT, InfershapeMaxPoolV3Grad_001) {
  ge::op::MaxPoolV3Grad op;
  auto input_desc = create_desc({2, 2}, ge::DT_FLOAT16);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("out_grad");
  std::vector<int64_t> expect_output_shape = {2, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expect_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

TEST_F(MaxPoolV3Grad_UT, InfershapeMaxPoolV3Grad_002) {
  ge::op::MaxPoolV3Grad op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}, {133, 133}};
  auto input_desc = create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {-1, -1}, ge::FORMAT_NCHW, range_x1);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("out_grad");
  std::vector<int64_t> expect_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expect_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{30, 30}, {133, 133}};
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(MaxPoolV3Grad_UT, InfershapeMaxPoolV3Grad_003) {
  ge::op::MaxPoolV3Grad op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}, {133, 133}};
  auto input_desc = create_desc_shape_range({2, -1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {2, -1}, ge::FORMAT_NCHW, range_x1);
  op.UpdateInputDesc("orig_input", input_desc);
  op.UpdateInputDesc("orig_output", input_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("out_grad");
  std::vector<int64_t> expect_output_shape = {2, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expect_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{30, 30}, {133, 133}};
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

