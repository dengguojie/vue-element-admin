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

class DynamicLSTMTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DynamicLSTMTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DynamicLSTMTest TearDown" << std::endl;
  }
};

TEST_F(DynamicLSTMTest, InfershapeDynamicLSTM_000) {
  ge::op::DynamicLSTM op;
  op.UpdateInputDesc("x", create_desc({4, 3}, ge::DT_INT8));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DynamicLSTMTest, InfershapeDynamicLSTM_001) {
  ge::op::DynamicLSTM op;
  op.UpdateInputDesc("x", create_desc({4, 4, 3}, ge::DT_FLOAT16));
  op.UpdateInputDesc("w", create_desc({4, 4, 3}, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_h_desc = op.GetOutputDesc("output_h");
  std::vector<int64_t> expected_output_shape = {4, 4, 1};
  EXPECT_EQ(output_h_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_h_desc.GetDataType(), ge::DT_FLOAT16);
}