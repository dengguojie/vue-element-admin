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

class LambApplyWeightAssign_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "LambApplyWeightAssign_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LambApplyWeightAssign_UT TearDown" << std::endl;
  }
};

TEST_F(LambApplyWeightAssign_UT, InfershapeLambApplyWeightAssign_001) {
  ge::op::LambApplyWeightAssign op;
  op.UpdateInputDesc("input_param", create_desc({4, 3, 1}, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc_output0 = op.GetOutputDesc("input_param");
  EXPECT_EQ(output_desc_output0.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_desc_output0 = {4, 3, 1};
  EXPECT_EQ(output_desc_output0.GetShape().GetDims(), expected_output_desc_output0);
}
