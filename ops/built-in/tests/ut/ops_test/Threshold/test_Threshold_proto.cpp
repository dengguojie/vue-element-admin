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

class Threshold_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Threshold_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Threshold_UT TearDown" << std::endl;
  }
};

TEST_F(Threshold_UT, InfershapeThreshold_test_01) {
  ge::op::Threshold op;
  op.UpdateInputDesc("x", create_desc_with_ori({4, 3, 1}, ge::DT_FLOAT16, 
                     ge::FORMAT_NCHW, {4, 3, 1}, ge::FORMAT_NCHW));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4, 3, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}