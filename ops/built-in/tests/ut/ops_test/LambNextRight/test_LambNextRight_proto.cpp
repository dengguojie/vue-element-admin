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

class LambNextRight_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "LambNextRight_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LambNextRight_UT TearDown" << std::endl;
  }
};

TEST_F(LambNextRight_UT, InfershapeLambNextRight_001) {
  ge::op::LambNextRight op;
  op.UpdateInputDesc("input_square", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("input_mul2", create_desc({4, 3, 1}, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
