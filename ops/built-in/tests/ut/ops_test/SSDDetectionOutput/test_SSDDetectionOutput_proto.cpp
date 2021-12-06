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

class SSDDetectionOutputTest_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SSDDetectionOutputTest_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SSDDetectionOutputTest_UT TearDown" << std::endl;
  }
};

TEST_F(SSDDetectionOutputTest_UT, InferShapeSSDDetectionOutput_000) {
  ge::op::SSDDetectionOutput op;
  op.UpdateInputDesc("bbox_delta", create_desc({}, ge::DT_INT8));
  op.SetAttr("top_k", 1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SSDDetectionOutputTest_UT, InferShapeSSDDetectionOutput_001) {
  ge::op::SSDDetectionOutput op;
  op.UpdateInputDesc("bbox_delta", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.SetAttr("top_k", -1);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto boxnum_output_desc = op.GetOutputDesc("out_boxnum");
  std::vector<int64_t> expected_output_shape_boxnum = {4, 8};
  EXPECT_EQ(boxnum_output_desc.GetShape().GetDims(), expected_output_shape_boxnum);
  EXPECT_EQ(boxnum_output_desc.GetDataType(), ge::DT_INT32);

  auto output_y_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_y_shape = {4, 1024, 8};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_y_shape);
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);
}