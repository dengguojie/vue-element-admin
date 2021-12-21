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

class BasicLSTMCellInputGradTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BasicLSTMCellInputGradTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BasicLSTMCellInputGradTest TearDown" << std::endl;
  }
};

TEST_F(BasicLSTMCellInputGradTest, InfershapeBasicLSTMCellInputGrad_000) {
  ge::op::BasicLSTMCellInputGrad op;
  op.UpdateInputDesc("dgate", create_desc({4, 3, 1}, ge::DT_INT8));
  op.UpdateInputDesc("w", create_desc({4, 3, 1}, ge::DT_INT8));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(BasicLSTMCellInputGradTest, InfershapeBasicLSTMCellInputGrad_001) {
  ge::op::BasicLSTMCellInputGrad op;
  op.UpdateInputDesc("dgate", create_desc({4, 3}, ge::DT_INT8));
  op.UpdateInputDesc("w", create_desc({4, 3, 1}, ge::DT_INT8));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(BasicLSTMCellInputGradTest, InfershapeBasicLSTMCellInputGrad_002) {
  ge::op::BasicLSTMCellInputGrad op;
  op.UpdateInputDesc("dgate", create_desc({4, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("w", create_desc({4, 4}, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_dxt_desc = op.GetOutputDesc("dxt");
  std::vector<int64_t> expected_output_shape_dxt = {4, 3};
  EXPECT_EQ(output_dxt_desc.GetShape().GetDims(), expected_output_shape_dxt);
  EXPECT_EQ(output_dxt_desc.GetDataType(), ge::DT_FLOAT16);

  auto output_dht_desc = op.GetOutputDesc("dht");
  std::vector<int64_t> expected_output_shape_dht = {4, 1};
  EXPECT_EQ(output_dht_desc.GetShape().GetDims(), expected_output_shape_dht);
  EXPECT_EQ(output_dht_desc.GetDataType(), ge::DT_FLOAT16);
}