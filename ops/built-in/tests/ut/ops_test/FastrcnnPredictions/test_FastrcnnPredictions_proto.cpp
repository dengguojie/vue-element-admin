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

class FastrcnnPredictionsTest_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FastrcnnPredictionsTest_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FastrcnnPredictionsTest_UT TearDown" << std::endl;
  }
};

TEST_F(FastrcnnPredictionsTest_UT, InferShapeFastrcnnPredictions_000) {
  ge::op::FastrcnnPredictions op;
  op.UpdateInputDesc("score",
                     create_desc_with_ori({4, 3, 1}, ge::DT_INT8, ge::FORMAT_NC1HWC0, {4, 3, 1}, ge::FORMAT_NC1HWC0));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FastrcnnPredictionsTest_UT, InferShapeFastrcnnPredictions_001) {
  ge::op::FastrcnnPredictions op;
  op.UpdateInputDesc("score", create_desc_with_ori({}, ge::DT_INT8, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  float nms_threshold = 0.1;
  float score_threshold = 0.1;
  op.SetAttr("nms_threshold", nms_threshold);
  op.SetAttr("score_threshold", score_threshold);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FastrcnnPredictionsTest_UT, InferShapeFastrcnnPredictions_002) {
  ge::op::FastrcnnPredictions op;
  auto input_desc = create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3, 1}, ge::FORMAT_ND);
  op.UpdateInputDesc("score", input_desc);
  op.SetAttr("nms_threshold", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FastrcnnPredictionsTest_UT, InferShapeFastrcnnPredictions_003) {
  ge::op::FastrcnnPredictions op;
  auto input_desc = create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3, 1}, ge::FORMAT_ND);
  op.UpdateInputDesc("score", input_desc);
  float nms_threshold = 10;
  op.SetAttr("nms_threshold", nms_threshold);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FastrcnnPredictionsTest_UT, InferShapeFastrcnnPredictions_004) {
  ge::op::FastrcnnPredictions op;
  auto input_desc = create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3, 1}, ge::FORMAT_ND);
  op.UpdateInputDesc("score", input_desc);
  float nms_threshold = 0.1;
  op.SetAttr("nms_threshold", nms_threshold);
  op.SetAttr("score_threshold", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FastrcnnPredictionsTest_UT, InferShapeFastrcnnPredictions_005) {
  ge::op::FastrcnnPredictions op;
  auto input_desc = create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3, 1}, ge::FORMAT_ND);
  op.UpdateInputDesc("score", input_desc);
  float nms_threshold = 0.1;
  float score_threshold = 10;
  op.SetAttr("nms_threshold", nms_threshold);
  op.SetAttr("score_threshold", score_threshold);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FastrcnnPredictionsTest_UT, InferShapeFastrcnnPredictions_006) {
  ge::op::FastrcnnPredictions op;
  op.UpdateInputDesc("score", create_desc_with_ori({4, 4}, ge::DT_INT8, ge::FORMAT_ND, {4, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("rois", create_desc_with_ori({}, ge::DT_INT8, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  float nms_threshold = 0.1;
  float score_threshold = 0.1;
  op.SetAttr("nms_threshold", nms_threshold);
  op.SetAttr("score_threshold", score_threshold);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FastrcnnPredictionsTest_UT, InferShapeFastrcnnPredictions_007) {
  ge::op::FastrcnnPredictions op;
  op.UpdateInputDesc("score", create_desc_with_ori({4, 4}, ge::DT_INT8, ge::FORMAT_ND, {4, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("rois", create_desc_with_ori({4, 3, 2}, ge::DT_INT8, ge::FORMAT_ND, {4, 3, 2}, ge::FORMAT_ND));
  float nms_threshold = 0.1;
  float score_threshold = 0.1;
  op.SetAttr("nms_threshold", nms_threshold);
  op.SetAttr("score_threshold", score_threshold);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FastrcnnPredictionsTest_UT, InferShapeFastrcnnPredictions_008) {
  ge::op::FastrcnnPredictions op;
  op.UpdateInputDesc("score", create_desc_with_ori({4, 4}, ge::DT_INT8, ge::FORMAT_ND, {4, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("rois", create_desc_with_ori({4, 3}, ge::DT_INT8, ge::FORMAT_ND, {4, 3}, ge::FORMAT_ND));
  float nms_threshold = 0.1;
  float score_threshold = 0.1;
  op.SetAttr("nms_threshold", nms_threshold);
  op.SetAttr("score_threshold", score_threshold);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FastrcnnPredictionsTest_UT, InferShapeFastrcnnPredictions_009) {
  ge::op::FastrcnnPredictions op;
  op.UpdateInputDesc("score", create_desc_with_ori({32, 64}, ge::DT_INT8, ge::FORMAT_ND, {32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("rois", create_desc_with_ori({4, 3}, ge::DT_INT8, ge::FORMAT_ND, {4, 3}, ge::FORMAT_ND));
  float nms_threshold = 0.1;
  float score_threshold = 0.1;
  op.SetAttr("nms_threshold", nms_threshold);
  op.SetAttr("score_threshold", score_threshold);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FastrcnnPredictionsTest_UT, InferShapeFastrcnnPredictions_010) {
  ge::op::FastrcnnPredictions op;
  op.UpdateInputDesc("score", create_desc_with_ori({32, 4}, ge::DT_INT8, ge::FORMAT_ND, {32, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("rois", create_desc_with_ori({4, 3}, ge::DT_INT8, ge::FORMAT_ND, {4, 3}, ge::FORMAT_ND));
  float nms_threshold = 0.1;
  float score_threshold = 0.1;
  op.SetAttr("nms_threshold", nms_threshold);
  op.SetAttr("score_threshold", score_threshold);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FastrcnnPredictionsTest_UT, InferShapeFastrcnnPredictions_011) {
  ge::op::FastrcnnPredictions op;
  op.UpdateInputDesc("score", create_desc_with_ori({32, 4}, ge::DT_INT8, ge::FORMAT_ND, {32, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("rois", create_desc_with_ori({4, 4}, ge::DT_INT8, ge::FORMAT_ND, {4, 4}, ge::FORMAT_ND));
  float nms_threshold = 0.1;
  float score_threshold = 0.1;
  op.SetAttr("nms_threshold", nms_threshold);
  op.SetAttr("score_threshold", score_threshold);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FastrcnnPredictionsTest_UT, InferShapeFastrcnnPredictions_012) {
  ge::op::FastrcnnPredictions op;
  op.UpdateInputDesc("score", create_desc_with_ori({16, 4}, ge::DT_INT8, ge::FORMAT_ND, {16, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("rois", create_desc_with_ori({48, 4}, ge::DT_INT8, ge::FORMAT_ND, {48, 4}, ge::FORMAT_ND));
  float nms_threshold = 0.1;
  float score_threshold = 0.1;
  op.SetAttr("nms_threshold", nms_threshold);
  op.SetAttr("score_threshold", score_threshold);
  op.SetAttr("k", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FastrcnnPredictionsTest_UT, InferShapeFastrcnnPredictions_013) {
  ge::op::FastrcnnPredictions op;
  op.UpdateInputDesc("score", create_desc_with_ori({16, 4}, ge::DT_INT8, ge::FORMAT_ND, {16, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("rois", create_desc_with_ori({48, 4}, ge::DT_INT8, ge::FORMAT_ND, {48, 4}, ge::FORMAT_ND));
  float nms_threshold = 0.1;
  float score_threshold = 0.1;
  op.SetAttr("nms_threshold", nms_threshold);
  op.SetAttr("score_threshold", score_threshold);
  op.SetAttr("k", 10);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(FastrcnnPredictionsTest_UT, InferShapeFastrcnnPredictions_014) {
  ge::op::FastrcnnPredictions op;
  op.UpdateInputDesc("score", create_desc_with_ori({16, 4}, ge::DT_INT8, ge::FORMAT_ND, {16, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("rois", create_desc_with_ori({48, 4}, ge::DT_INT8, ge::FORMAT_ND, {48, 4}, ge::FORMAT_ND));
  float nms_threshold = 0.1;
  float score_threshold = 0.1;
  int64_t k_num = 16;
  op.SetAttr("nms_threshold", nms_threshold);
  op.SetAttr("score_threshold", score_threshold);
  op.SetAttr("k", k_num);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto rois_output_desc = op.GetOutputDesc("sorted_rois");
  std::vector<int64_t> expected_output_shape_rois = {16, 4};
  EXPECT_EQ(rois_output_desc.GetShape().GetDims(), expected_output_shape_rois);

  std::vector<int64_t> expected_output_shape = {16, 1};
  auto scores_output_desc = op.GetOutputDesc("sorted_scores");
  auto classes_output_desc = op.GetOutputDesc("sorted_classes");
  EXPECT_EQ(scores_output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(classes_output_desc.GetShape().GetDims(), expected_output_shape);
}