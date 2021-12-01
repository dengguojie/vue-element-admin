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

class DecodeBboxTest_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DecodeBboxTest_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DecodeBboxTest_UT TearDown" << std::endl;
  }
};

TEST_F(DecodeBboxTest_UT, InferShapeDecodeBbox_000) {
  ge::op::DecodeBbox op;
  op.UpdateInputDesc("box_predictions",
                     create_desc_with_ori({4, 3, 1}, ge::DT_INT8, ge::FORMAT_NC1HWC0, {4, 3, 1}, ge::FORMAT_NC1HWC0));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxTest_UT, InferShapeDecodeBbox_001) {
  ge::op::DecodeBbox op;
  op.UpdateInputDesc("box_predictions",
                     create_desc_with_ori({4, 3, 1}, ge::DT_INT8, ge::FORMAT_NCHW, {4, 3, 1}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("anchors",
                     create_desc_with_ori({4, 3, 1}, ge::DT_INT8, ge::FORMAT_NC1HWC0, {4, 3, 1}, ge::FORMAT_NC1HWC0));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxTest_UT, InferShapeDecodeBbox_002) {
  ge::op::DecodeBbox op;
  auto input_desc = create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 3, 1}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("box_predictions", input_desc);
  op.UpdateInputDesc("anchors", input_desc);
  op.SetAttr("decode_clip", "true");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxTest_UT, InferShapeDecodeBbox_003) {
  ge::op::DecodeBbox op;
  auto input_desc = create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 3, 1}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("box_predictions", input_desc);
  op.UpdateInputDesc("anchors", input_desc);
  float decode_clip = -8;
  op.SetAttr("decode_clip", decode_clip);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxTest_UT, InferShapeDecodeBbox_004) {
  ge::op::DecodeBbox op;
  auto input_desc = create_desc_with_ori({}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("box_predictions", input_desc);
  op.UpdateInputDesc("anchors", input_desc);
  float decode_clip = 1.0;
  op.SetAttr("decode_clip", decode_clip);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxTest_UT, InferShapeDecodeBbox_005) {
  ge::op::DecodeBbox op;
  op.UpdateInputDesc("box_predictions",
                     create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 3, 1}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("anchors", create_desc_with_ori({}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {}, ge::FORMAT_NCHW));
  float decode_clip = 1.0;
  op.SetAttr("decode_clip", decode_clip);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxTest_UT, InferShapeDecodeBbox_006) {
  ge::op::DecodeBbox op;
  op.UpdateInputDesc("box_predictions",
                     create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 3, 1}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("anchors",
                     create_desc_with_ori({1, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1, 1}, ge::FORMAT_NCHW));
  float decode_clip = 1.0;
  op.SetAttr("decode_clip", decode_clip);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxTest_UT, InferShapeDecodeBbox_007) {
  ge::op::DecodeBbox op;
  op.UpdateInputDesc("box_predictions",
                     create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 3, 1}, ge::FORMAT_NCHW));
  op.UpdateInputDesc(
      "anchors", create_desc_with_ori({4, 3, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 3, 1, 1}, ge::FORMAT_NCHW));
  float decode_clip = 1.0;
  op.SetAttr("decode_clip", decode_clip);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxTest_UT, InferShapeDecodeBbox_008) {
  ge::op::DecodeBbox op;
  op.UpdateInputDesc("box_predictions",
                     create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 3, 1}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("anchors",
                     create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 3, 1}, ge::FORMAT_NCHW));
  float decode_clip = 1.0;
  op.SetAttr("decode_clip", decode_clip);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxTest_UT, InferShapeDecodeBbox_009) {
  ge::op::DecodeBbox op;
  op.UpdateInputDesc("box_predictions", create_desc_with_ori({4, 3, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                             {4, 3, 1, 1}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("anchors",
                     create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 3, 1}, ge::FORMAT_NCHW));
  float decode_clip = 1.0;
  op.SetAttr("decode_clip", decode_clip);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxTest_UT, InferShapeDecodeBbox_010) {
  ge::op::DecodeBbox op;
  op.UpdateInputDesc("box_predictions", create_desc_with_ori({2, 3, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                             {2, 3, 1, 1}, ge::FORMAT_NCHW));
  op.UpdateInputDesc(
      "anchors", create_desc_with_ori({2, 3, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 3, 1, 1}, ge::FORMAT_NCHW));
  float decode_clip = 1.0;
  op.SetAttr("decode_clip", decode_clip);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeBboxTest_UT, InferShapeDecodeBbox_011) {
  ge::op::DecodeBbox op;
  auto input_desc =
      create_desc_with_ori({4, 1, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 1, 1, 16}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("box_predictions", input_desc);
  op.UpdateInputDesc("anchors", input_desc);
  float decode_clip = 8.0;
  op.SetAttr("decode_clip", decode_clip);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto decoded_boxes_desc = op.GetOutputDesc("decoded_boxes");
  std::vector<int64_t> expected_output_shape = {4, 16};
  EXPECT_EQ(decoded_boxes_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DecodeBboxTest_UT, InferShapeDecodeBbox_012) {
  ge::op::DecodeBbox op;
  auto input_desc = create_desc_with_ori({6, 16, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6, 16, 4}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("box_predictions", input_desc);
  op.UpdateInputDesc("anchors", input_desc);
  float decode_clip = 8.0;
  op.SetAttr("decode_clip", decode_clip);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto decoded_boxes_desc = op.GetOutputDesc("decoded_boxes");
  std::vector<int64_t> expected_output_shape = {96, 4};
  EXPECT_EQ(decoded_boxes_desc.GetShape().GetDims(), expected_output_shape);
}