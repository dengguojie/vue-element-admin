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

class DynamicGRUCellGradTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DynamicGRUCellGradTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DynamicGRUCellGradTest TearDown" << std::endl;
  }
};

TEST_F(DynamicGRUCellGradTest, InfershapeDynamicGRUCellGrad_000) {
  ge::op::DynamicGRUCellGrad op;
  op.UpdateInputDesc("dy", create_desc({4, 3}, ge::DT_INT8));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DynamicGRUCellGradTest, InfershapeDynamicGRUCellGrad_001) {
  ge::op::DynamicGRUCellGrad op;
  op.UpdateInputDesc("dy", create_desc({4, 3, 1}, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto prev_output_desc = op.GetOutputDesc("dh_prev");
  std::vector<int64_t> expected_output_shape_prev = {1, 3, 1};
  EXPECT_EQ(prev_output_desc.GetShape().GetDims(), expected_output_shape_prev);
  EXPECT_EQ(prev_output_desc.GetDataType(), ge::DT_FLOAT16);

  auto dgate_output_desc = op.GetOutputDesc("dgate_h");
  std::vector<int64_t> expected_output_shape_dgate = {1, 3, 3};
  EXPECT_EQ(dgate_output_desc.GetShape().GetDims(), expected_output_shape_dgate);
  EXPECT_EQ(dgate_output_desc.GetDataType(), ge::DT_FLOAT16);

  auto dnt_output_desc = op.GetOutputDesc("dnt_x");
  std::vector<int64_t> expected_output_shape_dnt = {1, 3, 1};
  EXPECT_EQ(dnt_output_desc.GetShape().GetDims(), expected_output_shape_dnt);
  EXPECT_EQ(dnt_output_desc.GetDataType(), ge::DT_FLOAT16);
}