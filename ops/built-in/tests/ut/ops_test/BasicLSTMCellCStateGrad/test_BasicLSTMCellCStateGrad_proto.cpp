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
#include "rnn.h"
#include "op_proto_test_util.h"

class BasicLSTMCellCStateGradTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BasicLSTMCellCStateGradTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BasicLSTMCellCStateGradTest TearDown" << std::endl;
  }
};

TEST_F(BasicLSTMCellCStateGradTest, InfershapeBasicLSTMCellCStateGrad_000) {
  ge::op::BasicLSTMCellCStateGrad op;
  op.UpdateInputDesc("c", create_desc({4, 3, 1}, ge::DT_INT8));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(BasicLSTMCellCStateGradTest, InfershapeBasicLSTMCellCStateGrad_001) {
  ge::op::BasicLSTMCellCStateGrad op;
  op.UpdateInputDesc("c", create_desc({4, 1}, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_dgate_desc = op.GetOutputDesc("dgate");
  std::vector<int64_t> expected_output_shape_dgate = {4, 4};
  EXPECT_EQ(output_dgate_desc.GetShape().GetDims(), expected_output_shape_dgate);
  EXPECT_EQ(output_dgate_desc.GetDataType(), ge::DT_FLOAT16);
  
  auto output_dct_desc = op.GetOutputDesc("dct_1");
  std::vector<int64_t> expected_output_shape_dct = {4, 1};
  EXPECT_EQ(output_dct_desc.GetShape().GetDims(), expected_output_shape_dct);
  EXPECT_EQ(output_dct_desc.GetDataType(), ge::DT_FLOAT16);
}