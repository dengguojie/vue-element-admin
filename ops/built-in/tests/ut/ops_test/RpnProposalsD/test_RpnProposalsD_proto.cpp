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
#include <vector>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class RpnProposalsDTest_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RpnProposalsDTest_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RpnProposalsDTest_UT TearDown" << std::endl;
  }
};

TEST_F(RpnProposalsDTest_UT, InferShapeRpnProposalsD_000) {
  ge::op::RpnProposalsD op;
  auto input_desc = create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 3, 1}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("rois", input_desc);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(RpnProposalsDTest_UT, InferShapeRpnProposalsD_001) {
  ge::op::RpnProposalsD op;
  auto input_desc = create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3, 1}, ge::FORMAT_ND);
  op.UpdateInputDesc("rois", input_desc);
  op.UpdateInputDesc("cls_bg_prob", create_desc_with_ori({}, ge::DT_FLOAT16, ge::FORMAT_ND, {}, ge::FORMAT_ND));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(RpnProposalsDTest_UT, InferShapeRpnProposalsD_002) {
  ge::op::RpnProposalsD op;
  auto input_desc = create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3, 1}, ge::FORMAT_ND);
  op.UpdateInputDesc("cls_bg_prob", input_desc);
  op.UpdateInputDesc("rois", create_desc_with_ori({}, ge::DT_FLOAT16, ge::FORMAT_ND, {}, ge::FORMAT_ND));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(RpnProposalsDTest_UT, InferShapeRpnProposalsD_003) {
  ge::op::RpnProposalsD op;
  auto input_desc = create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3, 1}, ge::FORMAT_ND);
  op.UpdateInputDesc("cls_bg_prob", input_desc);
  op.UpdateInputDesc("rois", input_desc);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(RpnProposalsDTest_UT, InferShapeRpnProposalsD_004) {
  ge::op::RpnProposalsD op;
  auto input_desc = create_desc_with_ori({3, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {3, 3}, ge::FORMAT_ND);
  op.UpdateInputDesc("cls_bg_prob", input_desc);
  op.UpdateInputDesc("rois", input_desc);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(RpnProposalsDTest_UT, InferShapeRpnProposalsD_005) {
  ge::op::RpnProposalsD op;
  auto input_desc = create_desc_with_ori({4, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3}, ge::FORMAT_ND);
  op.UpdateInputDesc("cls_bg_prob", input_desc);
  op.UpdateInputDesc("rois", input_desc);
  op.SetAttr("score_threshold", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(RpnProposalsDTest_UT, InferShapeRpnProposalsD_006) {
  ge::op::RpnProposalsD op;
  auto input_desc = create_desc_with_ori({4, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3}, ge::FORMAT_ND);
  op.UpdateInputDesc("cls_bg_prob", input_desc);
  op.UpdateInputDesc("rois", input_desc);
  op.SetAttr("score_threshold", 0);
  op.SetAttr("nms_threshold", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(RpnProposalsDTest_UT, InferShapeRpnProposalsD_007) {
  ge::op::RpnProposalsD op;
  auto input_desc = create_desc_with_ori({4, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3}, ge::FORMAT_ND);
  op.UpdateInputDesc("cls_bg_prob", input_desc);
  op.UpdateInputDesc("rois", input_desc);
  op.SetAttr("score_threshold", 0);
  op.SetAttr("nms_threshold", 8);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(RpnProposalsDTest_UT, InferShapeRpnProposalsD_008) {
  ge::op::RpnProposalsD op;
  auto input_desc = create_desc_with_ori({4, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3}, ge::FORMAT_ND);
  op.UpdateInputDesc("cls_bg_prob", input_desc);
  op.UpdateInputDesc("rois", input_desc);
  op.SetAttr("score_threshold", 0);
  op.SetAttr("nms_threshold", 1);
  op.SetAttr("k", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(RpnProposalsDTest_UT, InferShapeRpnProposalsD_009) {
  ge::op::RpnProposalsD op;
  auto input_desc = create_desc_with_ori({4, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3}, ge::FORMAT_ND);
  op.UpdateInputDesc("cls_bg_prob", input_desc);
  op.UpdateInputDesc("rois", input_desc);
  op.SetAttr("score_threshold", 0);
  op.SetAttr("nms_threshold", 1);
  op.SetAttr("k", 1);
  op.SetAttr("min_size", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(RpnProposalsDTest_UT, InferShapeRpnProposalsD_010) {
  ge::op::RpnProposalsD op;
  auto input_desc = create_desc_with_ori({4, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3}, ge::FORMAT_ND);
  op.UpdateInputDesc("cls_bg_prob", input_desc);
  op.UpdateInputDesc("rois", input_desc);
  op.SetAttr("score_threshold", 0);
  op.SetAttr("nms_threshold", 1);
  op.SetAttr("k", 1);
  float min_size = -10.0;
  op.SetAttr("min_size", min_size);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(RpnProposalsDTest_UT, InferShapeRpnProposalsD_011) {
  ge::op::RpnProposalsD op;
  auto input_desc = create_desc_with_ori({4, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3}, ge::FORMAT_ND);
  op.UpdateInputDesc("cls_bg_prob", input_desc);
  op.UpdateInputDesc("rois", input_desc);
  op.SetAttr("score_threshold", 0);
  op.SetAttr("nms_threshold", 1);
  op.SetAttr("k", 1);
  float min_size = 1.0;
  op.SetAttr("min_size", min_size);
  op.SetAttr("img_size", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(RpnProposalsDTest_UT, InferShapeRpnProposalsD_012) {
  ge::op::RpnProposalsD op;
  auto input_desc = create_desc_with_ori({4, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3}, ge::FORMAT_ND);
  op.UpdateInputDesc("cls_bg_prob", input_desc);
  op.UpdateInputDesc("rois", input_desc);
  op.SetAttr("score_threshold", 0);
  op.SetAttr("nms_threshold", 1);
  op.SetAttr("k", 1);
  float min_size = 1.0;
  op.SetAttr("min_size", min_size);
  std::vector<int64_t> img_size = {12, 4, 1};
  op.SetAttr("img_size", img_size);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(RpnProposalsDTest_UT, InferShapeRpnProposalsD_013) {
  ge::op::RpnProposalsD op;
  auto input_desc = create_desc_with_ori({4, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3}, ge::FORMAT_ND);
  op.UpdateInputDesc("cls_bg_prob", input_desc);
  op.UpdateInputDesc("rois", input_desc);
  op.SetAttr("score_threshold", 0);
  op.SetAttr("nms_threshold", 1);
  op.SetAttr("k", 1);
  float min_size = 1.0;
  op.SetAttr("min_size", min_size);
  std::vector<int64_t> img_size = {12, -1};
  op.SetAttr("img_size", img_size);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(RpnProposalsDTest_UT, InferShapeRpnProposalsD_014) {
  ge::op::RpnProposalsD op;
  op.SetAttr("post_nms_num", 12);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto box_output_desc = op.GetOutputDesc("sorted_box");
  std::vector<int64_t> expected_output_shape = {12, 4};
  EXPECT_EQ(box_output_desc.GetShape().GetDims(), expected_output_shape);
}