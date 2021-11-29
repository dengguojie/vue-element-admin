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

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class AddV2_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AddV2_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AddV2_UT TearDown" << std::endl;
  }
};

TEST_F(AddV2_UT, InferShapeAddV2_000) {
  ge::op::AddV2 op;
  op.UpdateInputDesc("x1", create_desc_with_ori({1, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("x2", create_desc_with_ori({1, 2}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 2}, ge::FORMAT_NHWC));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AddV2_UT, InferShapeAddV2_001) {
  ge::op::AddV2 op;
  op.UpdateInputDesc("x1", create_desc_with_ori({1, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("x2", create_desc_with_ori({1, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC, {1, 2}, ge::FORMAT_NHWC));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}