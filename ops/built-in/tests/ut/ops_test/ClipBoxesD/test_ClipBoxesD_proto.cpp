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

class ClipBoxesDTest_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ClipBoxesDTest_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ClipBoxesDTest_UT TearDown" << std::endl;
  }
};

TEST_F(ClipBoxesDTest_UT, InferShapeClipBoxesD_000) {
  ge::op::ClipBoxesD op;
  op.UpdateInputDesc("boxes_input",
                     create_desc_with_ori({4, 3, 1}, ge::DT_INT8, ge::FORMAT_NC1HWC0, {4, 3, 1}, ge::FORMAT_NC1HWC0));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ClipBoxesDTest_UT, InferShapeClipBoxesD_001) {
  ge::op::ClipBoxesD op;
  op.UpdateInputDesc("boxes_input", create_desc_with_ori({}, ge::DT_INT8, ge::FORMAT_ND, {}, ge::FORMAT_ND));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ClipBoxesDTest_UT, InferShapeClipBoxesD_002) {
  ge::op::ClipBoxesD op;
  auto input_desc = create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3, 1}, ge::FORMAT_ND);
  op.UpdateInputDesc("boxes_input", input_desc);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ClipBoxesDTest_UT, InferShapeClipBoxesD_003) {
  ge::op::ClipBoxesD op;
  auto input_desc = create_desc_with_ori({65501, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {65501, 1}, ge::FORMAT_ND);
  op.UpdateInputDesc("boxes_input", input_desc);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ClipBoxesDTest_UT, InferShapeClipBoxesD_004) {
  ge::op::ClipBoxesD op;
  auto input_desc = create_desc_with_ori({4, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 1}, ge::FORMAT_ND);
  op.UpdateInputDesc("boxes_input", input_desc);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ClipBoxesDTest_UT, InferShapeClipBoxesD_005) {
  ge::op::ClipBoxesD op;
  auto input_desc = create_desc_with_ori({4000, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {4000, 4}, ge::FORMAT_ND);
  op.UpdateInputDesc("boxes_input", input_desc);
  op.SetAttr("img_size", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ClipBoxesDTest_UT, InferShapeClipBoxesD_006) {
  ge::op::ClipBoxesD op;
  auto input_desc = create_desc_with_ori({4000, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {4000, 4}, ge::FORMAT_ND);
  op.UpdateInputDesc("boxes_input", input_desc);
  std::vector<int64_t> img_size = {1024, 728, 1};
  op.SetAttr("img_size", img_size);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ClipBoxesDTest_UT, InferShapeClipBoxesD_007) {
  ge::op::ClipBoxesD op;
  auto input_desc = create_desc_with_ori({4000, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {4000, 4}, ge::FORMAT_ND);
  op.UpdateInputDesc("boxes_input", input_desc);
  std::vector<int64_t> img_size = {-1, 728};
  op.SetAttr("img_size", img_size);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ClipBoxesDTest_UT, InferShapeClipBoxesD_008) {
  ge::op::ClipBoxesD op;
  auto input_desc = create_desc_with_ori({4000, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {4000, 4}, ge::FORMAT_ND);
  op.UpdateInputDesc("boxes_input", input_desc);
  std::vector<int64_t> img_size = {1024, 728};
  op.SetAttr("img_size", img_size);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto boxes_output_desc = op.GetOutputDesc("boxes_output");
  std::vector<int64_t> expected_output_shape = {4000, 4};
  EXPECT_EQ(boxes_output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ClipBoxesDTest_UT, InferShapeClipBoxesD_009) {
  ge::op::ClipBoxesD op;
  auto input_desc = create_desc_with_ori({-1, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {4000, 4}, ge::FORMAT_ND);
  op.UpdateInputDesc("boxes_input", input_desc);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}