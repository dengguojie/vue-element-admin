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

#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

class AvgPoolV2GradD_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPoolV2GradD_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPoolV2GradD_UT TearDown" << std::endl;
  }
};
TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_001) {
  ge::op::AvgPoolV2GradD op;
  op.SetAttr("orig_input_shape", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_002) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  op.SetAttr("ksize", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_003) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  std::vector<int64_t> ksizeList = {5, 5, 5};
  op.SetAttr("ksize", ksizeList);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_004) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  std::vector<int64_t> ksizeList = {5, 5, 5, 5};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_005) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  std::vector<int64_t> ksizeList = {5, 5, 5, 5};
  std::vector<int64_t> stridesList = {3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_006) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  std::vector<int64_t> ksizeList = {1, 5, 5, 1};
  std::vector<int64_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", ksizeList);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_007) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  std::vector<int64_t> ksizeList = {1, 5, 5, 1};
  std::vector<int64_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "error");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_008) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  std::vector<int64_t> ksizeList = {1, 5, 5, 1};
  std::vector<int64_t> stridesList = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "VALID");
  op.SetAttr("pads", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_009) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  std::vector<int64_t> ksizeList = {1, 5, 5, 1};
  std::vector<int64_t> stridesList = {1, 3, 3, 1};
  std::vector<int64_t> pads = {1, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "VALID");
  op.SetAttr("pads", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_010) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  std::vector<int64_t> ksizeList = {1, 5, 5, 1};
  std::vector<int64_t> stridesList = {1, 3, 3, 1};
  std::vector<int64_t> pads = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "VALID");
  op.SetAttr("pads", pads);
  op.SetAttr("ceil_mode", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_011) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  std::vector<int64_t> ksizeList = {1, 5, 5, 1};
  std::vector<int64_t> stridesList = {1, 3, 3, 1};
  std::vector<int64_t> pads = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "VALID");
  op.SetAttr("pads", pads);
  op.SetAttr("ceil_mode", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_012) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  std::vector<int64_t> ksizeList = {1, 5, 5, 1};
  std::vector<int64_t> stridesList = {1, 3, 3, 1};
  std::vector<int64_t> pads = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "VALID");
  op.SetAttr("pads", pads);
  op.SetAttr("ceil_mode", true);
  op.SetAttr("data_format", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_013) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  std::vector<int64_t> ksizeList = {1, 5, 5, 1};
  std::vector<int64_t> stridesList = {1, 3, 3, 1};
  std::vector<int64_t> pads = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "VALID");
  op.SetAttr("pads", pads);
  op.SetAttr("ceil_mode", true);
  op.SetAttr("data_format", "ND");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_014) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  std::vector<int64_t> ksizeList = {3, 5, 5, 1};
  std::vector<int64_t> stridesList = {1, 3, 3, 1};
  std::vector<int64_t> pads = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "VALID");
  op.SetAttr("pads", pads);
  op.SetAttr("ceil_mode", true);
  op.SetAttr("data_format", "NCHW");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_015) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  std::vector<int64_t> ksizeList = {1, 1, 5, 1};
  std::vector<int64_t> stridesList = {1, 1, 3, 1};
  std::vector<int64_t> pads = {6, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", pads);
  op.SetAttr("ceil_mode", true);
  op.SetAttr("data_format", "NCHW");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_016) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  std::vector<int64_t> ksizeList = {2, 5, 5, 1};
  std::vector<int64_t> stridesList = {1, 3, 3, 1};
  std::vector<int64_t> pads = {1, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", pads);
  op.SetAttr("ceil_mode", true);
  op.SetAttr("data_format", "NHWC");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, VerifyAvgPoolV2GradD_017) {
  ge::op::AvgPoolV2GradD op;
  std::vector<int64_t> orig_input_shape = {1, 1, 1, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  std::vector<int64_t> ksizeList = {1, 5, 5, 1};
  std::vector<int64_t> stridesList = {1, 3, 3, 1};
  std::vector<int64_t> pads = {6, 3, 3, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "CALCULATED");
  op.SetAttr("pads", pads);
  op.SetAttr("ceil_mode", true);
  op.SetAttr("data_format", "NHWC");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolV2GradD_UT, InfershapeAvgPoolV2GradD_001) {
  ge::op::AvgPoolV2GradD op;
  op.UpdateInputDesc("orig_input_shape", create_desc({2, 2}, ge::DT_INT32));
  op.UpdateInputDesc("input_grad", create_desc({2, 2}, ge::DT_INT32));
  std::vector<int64_t> orig_input_size = {6, 3, 3, 1};
  op.SetAttr("orig_input_shape", orig_input_size);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("out_grad");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {6, 3, 3, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}