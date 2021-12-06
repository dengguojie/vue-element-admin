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

class DecodeWheelsTargetTest_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DecodeWheelsTargetTest_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DecodeWheelsTargetTest_UT TearDown" << std::endl;
  }
};

TEST_F(DecodeWheelsTargetTest_UT, InferShapeDecodeWheelsTarget_000) {
  ge::op::DecodeWheelsTarget op;
  op.UpdateInputDesc("boundary_predictions", create_desc({}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeWheelsTargetTest_UT, InferShapeDecodeWheelsTarget_001) {
  ge::op::DecodeWheelsTarget op;
  op.UpdateInputDesc("boundary_predictions", create_desc({4, 3, 1}, ge::DT_INT8));
  op.UpdateInputDesc("anchors", create_desc({}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeWheelsTargetTest_UT, InferShapeDecodeWheelsTarget_002) {
  ge::op::DecodeWheelsTarget op;
  op.UpdateInputDesc("boundary_predictions", create_desc({4, 3, 1}, ge::DT_INT8));
  op.UpdateInputDesc("anchors", create_desc({4, 4}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeWheelsTargetTest_UT, InferShapeDecodeWheelsTarget_003) {
  ge::op::DecodeWheelsTarget op;
  op.UpdateInputDesc("boundary_predictions", create_desc({4, 3}, ge::DT_INT8));
  op.UpdateInputDesc("anchors", create_desc({4, 4}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeWheelsTargetTest_UT, InferShapeDecodeWheelsTarget_004) {
  ge::op::DecodeWheelsTarget op;
  op.UpdateInputDesc("boundary_predictions", create_desc({4, 8}, ge::DT_INT8));
  op.UpdateInputDesc("anchors", create_desc({4, 3}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeWheelsTargetTest_UT, InferShapeDecodeWheelsTarget_005) {
  ge::op::DecodeWheelsTarget op;
  op.UpdateInputDesc("boundary_predictions", create_desc({3, 8}, ge::DT_INT8));
  op.UpdateInputDesc("anchors", create_desc({4, 4}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(DecodeWheelsTargetTest_UT, InferShapeDecodeWheelsTarget_006) {
  ge::op::DecodeWheelsTarget op;
  op.UpdateInputDesc("boundary_predictions", create_desc({4, 8}, ge::DT_INT8));
  op.UpdateInputDesc("anchors", create_desc({4, 4}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto boundary_boxes_desc = op.GetOutputDesc("boundary_encoded");
  std::vector<int64_t> expected_output_shape = {4, 8};
  EXPECT_EQ(boundary_boxes_desc.GetShape().GetDims(), expected_output_shape);
}