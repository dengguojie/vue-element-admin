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

class MaxPoolGradWithArgmax_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaxPoolGradWithArgmax_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPoolGradWithArgmax_UT TearDown" << std::endl;
  }
};

TEST_F(MaxPoolGradWithArgmax_UT, VerifyMaxPoolGradWithArgmax_001) {
  ge::op::MaxPoolGradWithArgmax op;
  op.SetAttr("ksize", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmax_UT, VerifyMaxPoolGradWithArgmax_002) {
  ge::op::MaxPoolGradWithArgmax op;
  std::vector<int32_t> ksizeList = {5, 5, 5};
  op.SetAttr("ksize", ksizeList);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmax_UT, VerifyMaxPoolGradWithArgmax_003) {
  ge::op::MaxPoolGradWithArgmax op;
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmax_UT, VerifyMaxPoolGradWithArgmax_004) {
  ge::op::MaxPoolGradWithArgmax op;
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmax_UT, VerifyMaxPoolGradWithArgmax_005) {
  ge::op::MaxPoolGradWithArgmax op;
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 5};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmax_UT, VerifyMaxPoolGradWithArgmax_006) {
  ge::op::MaxPoolGradWithArgmax op;
  std::vector<int32_t> ksizeList = {1, 255, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmax_UT, VerifyMaxPoolGradWithArgmax_007) {
  ge::op::MaxPoolGradWithArgmax op;
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", ksizeList);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmax_UT, VerifyMaxPoolGradWithArgmax_008) {
  ge::op::MaxPoolGradWithArgmax op;
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "error");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmax_UT, VerifyMaxPoolGradWithArgmax_009) {
  ge::op::MaxPoolGradWithArgmax op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16}, ge::FORMAT_NDHWC));
  op.UpdateInputDesc("grad", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 2, 2, 2, 16},
                                                  ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradWithArgmax_UT, VerifyMaxPoolGradWithArgmax_010) {
  ge::op::MaxPoolGradWithArgmax op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16},
                                               ge::FORMAT_NDHWC));
  op.UpdateInputDesc("grad", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 2, 2, 2, 16},
                                                  ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {1, 5, 5, 1};
  std::vector<int32_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("data_format", "NCHW");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}