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

class DecodeBoundariesTargetTest_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DecodeBoundariesTargetTest_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DecodeBoundariesTargetTest_UT TearDown" << std::endl;
  }
};

TEST_F(DecodeBoundariesTargetTest_UT, InferShapeDecodeBoundariesTarget_000) {
  ge::op::DecodeBoundariesTarget op;
  op.UpdateInputDesc("boundary_predictions",
                     create_desc_with_ori({4, 3, 1}, ge::DT_INT8, ge::FORMAT_NC1HWC0, {4, 3, 1}, ge::FORMAT_NC1HWC0));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBoundariesTargetTest_UT, InferShapeDecodeBoundariesTarget_001) {
  ge::op::DecodeBoundariesTarget op;
  op.UpdateInputDesc("boundary_predictions",
                     create_desc_with_ori({4, 3, 1}, ge::DT_INT8, ge::FORMAT_ND, {4, 3, 1}, ge::FORMAT_ND));
  op.UpdateInputDesc("anchors",
                     create_desc_with_ori({4, 3, 1}, ge::DT_INT8, ge::FORMAT_NC1HWC0, {4, 3, 1}, ge::FORMAT_NC1HWC0));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBoundariesTargetTest_UT, InferShapeDecodeBoundariesTarget_002) {
  ge::op::DecodeBoundariesTarget op;
  op.UpdateInputDesc("boundary_predictions",
                     create_desc_with_ori({4}, ge::DT_INT8, ge::FORMAT_ND, {4}, ge::FORMAT_ND));
  op.UpdateInputDesc("anchors",
                     create_desc_with_ori({4}, ge::DT_INT8, ge::FORMAT_ND, {4}, ge::FORMAT_ND));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBoundariesTargetTest_UT, InferShapeDecodeBoundariesTarget_003) {
  ge::op::DecodeBoundariesTarget op;
  op.UpdateInputDesc("boundary_predictions",
                     create_desc_with_ori({4,1}, ge::DT_INT8, ge::FORMAT_ND, {4,1}, ge::FORMAT_ND));
  op.UpdateInputDesc("anchors",
                     create_desc_with_ori({4}, ge::DT_INT8, ge::FORMAT_ND, {4}, ge::FORMAT_ND));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBoundariesTargetTest_UT, InferShapeDecodeBoundariesTarget_004) {
  ge::op::DecodeBoundariesTarget op;
  op.UpdateInputDesc("boundary_predictions",
                     create_desc_with_ori({4,1}, ge::DT_INT8, ge::FORMAT_ND, {4,1}, ge::FORMAT_ND));
  op.UpdateInputDesc("anchors",
                     create_desc_with_ori({3,1}, ge::DT_INT8, ge::FORMAT_ND, {3,1}, ge::FORMAT_ND));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBoundariesTargetTest_UT, InferShapeDecodeBoundariesTarget_005) {
  ge::op::DecodeBoundariesTarget op;
  op.UpdateInputDesc("boundary_predictions",
                     create_desc_with_ori({4,2}, ge::DT_INT8, ge::FORMAT_ND, {4,2}, ge::FORMAT_ND));
  op.UpdateInputDesc("anchors",
                     create_desc_with_ori({4,3}, ge::DT_INT8, ge::FORMAT_ND, {4,3}, ge::FORMAT_ND));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBoundariesTargetTest_UT, InferShapeDecodeBoundariesTarget_006) {
  ge::op::DecodeBoundariesTarget op;
  op.UpdateInputDesc("boundary_predictions",
                     create_desc_with_ori({4,1}, ge::DT_INT8, ge::FORMAT_ND, {4,1}, ge::FORMAT_ND));
  op.UpdateInputDesc("anchors",
                     create_desc_with_ori({4,3}, ge::DT_INT8, ge::FORMAT_ND, {4,3}, ge::FORMAT_ND));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBoundariesTargetTest_UT, InferShapeDecodeBoundariesTarget_007) {
  ge::op::DecodeBoundariesTarget op;
  op.UpdateInputDesc("boundary_predictions",
                     create_desc_with_ori({4,1}, ge::DT_INT8, ge::FORMAT_ND, {4,1}, ge::FORMAT_ND));
  op.UpdateInputDesc("anchors",
                     create_desc_with_ori({4,4}, ge::DT_INT8, ge::FORMAT_ND, {4,4}, ge::FORMAT_ND));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto boundary_boxes_desc = op.GetOutputDesc("boundary_encoded");
  std::vector<int64_t> expected_output_shape = {4,1};
  EXPECT_EQ(boundary_boxes_desc.GetShape().GetDims(), expected_output_shape);
}