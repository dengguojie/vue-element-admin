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

class BasicRNNCellTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BasicRNNCellTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BasicRNNCellTest TearDown" << std::endl;
  }
};

TEST_F(BasicRNNCellTest, InfershapeBasicRNNCell_000) {
  ge::op::BasicRNNCell op;
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_INT8));
  op.SetAttr("num_output", "zero");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(BasicRNNCellTest, InfershapeBasicRNNCell_001) {
  ge::op::BasicRNNCell op;
  op.UpdateInputDesc("x", create_desc({4, 3, 1}, ge::DT_INT8));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(BasicRNNCellTest, InfershapeBasicRNNCell_002) {
  ge::op::BasicRNNCell op;
  op.UpdateInputDesc("x", create_desc({4, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bias_h", create_desc({4, 4}, ge::DT_FLOAT16));
  op.SetAttr("num_output", 4);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_ot_desc = op.GetOutputDesc("o_t");
  std::vector<int64_t> expected_output_shape_ot = {4, 4};
  EXPECT_EQ(output_ot_desc.GetShape().GetDims(), expected_output_shape_ot);
  EXPECT_EQ(output_ot_desc.GetDataType(), ge::DT_FLOAT16);

  auto output_ht_desc = op.GetOutputDesc("h_t");
  std::vector<int64_t> expected_output_shape_dht = {4, 4};
  EXPECT_EQ(output_ht_desc.GetShape().GetDims(), expected_output_shape_dht);
  EXPECT_EQ(output_ht_desc.GetDataType(), ge::DT_FLOAT16);
}