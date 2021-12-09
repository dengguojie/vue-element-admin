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
#include "nn_pooling_ops.h"

class FractionalAvgPool_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FractionalAvgPool_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FractionalAvgPool_UT TearDown" << std::endl;
  }
};

TEST_F(FractionalAvgPool_UT, InfershapeFractionalAvgPool_001) {
  ge::op::FractionalAvgPool op;
  op.UpdateInputDesc("x", create_desc({2, 2}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(FractionalAvgPool_UT, InfershapeFractionalAvgPool_002) {
  ge::op::FractionalAvgPool op;
  op.UpdateInputDesc("x", create_desc({2, 2, 2, 2}, ge::DT_FLOAT));
  std::vector<float> pooling_ratio = {1.0, 2.0};
  op.SetAttr("pooling_ratio", pooling_ratio);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(FractionalAvgPool_UT, InfershapeFractionalAvgPool_003) {
  ge::op::FractionalAvgPool op;
  op.UpdateInputDesc("x", create_desc({2, 2, 2, 2}, ge::DT_FLOAT));
  std::vector<float> pooling_ratio = {1.0, 1.0, 1.0, 1.0};
  op.SetAttr("pooling_ratio", pooling_ratio);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape_y = { 2, 2, 2, 2};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape_y);

  auto output_row_desc = op.GetOutputDesc("row_pooling_sequence");
  EXPECT_EQ(output_row_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_output_shape_row = {3};
  EXPECT_EQ(output_row_desc.GetShape().GetDims(), expected_output_shape_row);

  auto output_col_desc = op.GetOutputDesc("col_pooling_sequence");
  EXPECT_EQ(output_col_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_output_shape_col = {3};
  EXPECT_EQ(output_col_desc.GetShape().GetDims(), expected_output_shape_col);
}
