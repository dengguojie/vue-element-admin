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

class ClipBoxesTest_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ClipBoxesTest_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ClipBoxesTest_UT TearDown" << std::endl;
  }
};

TEST_F(ClipBoxesTest_UT, InferShapeClipBoxes_000) {
  ge::op::ClipBoxes op;
  op.UpdateInputDesc("boxes_input", create_desc({6, 16, 4}, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto boxes_output_desc = op.GetOutputDesc("boxes_output");
  std::vector<int64_t> expected_output_shape = {6, 16, 4};
  EXPECT_EQ(boxes_output_desc.GetShape().GetDims(), expected_output_shape);
}