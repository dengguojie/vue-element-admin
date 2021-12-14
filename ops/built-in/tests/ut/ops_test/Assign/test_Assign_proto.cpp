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
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class Assign_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Assign_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Assign_UT TearDown" << std::endl;
  }
};

TEST_F(Assign_UT, VerifyAssign_001) {
  ge::op::Assign op;
  op.UpdateInputDesc("ref", create_desc({2, 2}, ge::DT_FLOAT16));
  op.UpdateInputDesc("value", create_desc({2, 2}, ge::DT_FLOAT));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(Assign_UT, InfershapeAssign_001) {
  ge::op::Assign op;
  op.UpdateInputDesc("ref", create_desc({2, 2}, ge::DT_FLOAT16));
  op.UpdateInputDesc("value", create_desc({2, 2}, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}