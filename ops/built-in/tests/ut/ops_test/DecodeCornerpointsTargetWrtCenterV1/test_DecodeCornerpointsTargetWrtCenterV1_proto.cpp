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

class DecodeCornerpointsTargetWrtCenterV1Test_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DecodeCornerpointsTargetWrtCenterV1Test_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DecodeCornerpointsTargetWrtCenterV1Test_UT TearDown" << std::endl;
  }
};

TEST_F(DecodeCornerpointsTargetWrtCenterV1Test_UT, InferShapeDecodeCornerpointsTargetWrtCenterV1_000) {
  ge::op::DecodeCornerpointsTargetWrtCenterV1 op;
  op.UpdateInputDesc("keypoints_prediction", create_desc({}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeCornerpointsTargetWrtCenterV1Test_UT, InferShapeDecodeCornerpointsTargetWrtCenterV1_001) {
  ge::op::DecodeCornerpointsTargetWrtCenterV1 op;
  op.UpdateInputDesc("keypoints_prediction", create_desc({4, 3, 1}, ge::DT_INT8));
  op.UpdateInputDesc("anchors", create_desc({}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeCornerpointsTargetWrtCenterV1Test_UT, InferShapeDecodeCornerpointsTargetWrtCenterV1_002) {
  ge::op::DecodeCornerpointsTargetWrtCenterV1 op;
  op.UpdateInputDesc("keypoints_prediction", create_desc({4, 3, 1}, ge::DT_INT8));
  op.UpdateInputDesc("anchors", create_desc({4, 4}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeCornerpointsTargetWrtCenterV1Test_UT, InferShapeDecodeCornerpointsTargetWrtCenterV1_003) {
  ge::op::DecodeCornerpointsTargetWrtCenterV1 op;
  op.UpdateInputDesc("keypoints_prediction", create_desc({4, 3}, ge::DT_INT8));
  op.UpdateInputDesc("anchors", create_desc({4, 4}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeCornerpointsTargetWrtCenterV1Test_UT, InferShapeDecodeCornerpointsTargetWrtCenterV1_004) {
  ge::op::DecodeCornerpointsTargetWrtCenterV1 op;
  op.UpdateInputDesc("keypoints_prediction", create_desc({4, 8}, ge::DT_INT8));
  op.UpdateInputDesc("anchors", create_desc({4, 3}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeCornerpointsTargetWrtCenterV1Test_UT, InferShapeDecodeCornerpointsTargetWrtCenterV1_005) {
  ge::op::DecodeCornerpointsTargetWrtCenterV1 op;
  op.UpdateInputDesc("keypoints_prediction", create_desc({3, 8}, ge::DT_INT8));
  op.UpdateInputDesc("anchors", create_desc({4, 4}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeCornerpointsTargetWrtCenterV1Test_UT, InferShapeDecodeCornerpointsTargetWrtCenterV1_006) {
  ge::op::DecodeCornerpointsTargetWrtCenterV1 op;
  op.UpdateInputDesc("keypoints_prediction", create_desc({4, 8}, ge::DT_INT8));
  op.UpdateInputDesc("anchors", create_desc({4, 4}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto keypoints_boxes_desc = op.GetOutputDesc("keypoints_decoded");
  std::vector<int64_t> expected_output_shape = {4, 8};
  EXPECT_EQ(keypoints_boxes_desc.GetShape().GetDims(), expected_output_shape);
}