/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "selection_ops.h"

class inplace_update : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "inplace_update SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "inplace_update TearDown" << std::endl;
  }
};

TEST_F(inplace_update, inplace_update_infershape_diff_test_1) {
  ge::op::InplaceUpdate op;
  op.UpdateInputDesc("x", create_desc_shape_range({2, 2, 2},
                                                  ge::DT_FLOAT, ge::FORMAT_ND,
                                                  {2, 2, 2}, ge::FORMAT_ND,
                                                  {{2, 2}, {2, 2}, {2, 2}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("x");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {2, 2, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(inplace_update, inplace_update_infershape_diff_test_2) {
  ge::op::InplaceUpdate op;
  op.UpdateInputDesc("x", create_desc_shape_range({2, 2, -1},
                                                  ge::DT_FLOAT, ge::FORMAT_ND,
                                                  {2, 2, -1}, ge::FORMAT_ND,
                                                  {{2, 2}, {2, 2}, {2, 20}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("x");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {2, 2, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{2, 2}, {2, 2}, {2, 20}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(inplace_update, inplace_update_infershape_diff_test_3) {
  ge::op::InplaceUpdate op;
  op.UpdateInputDesc("x", create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {-2},
                                                  ge::FORMAT_ND, {{2, 2}, {2, 2}, {3, 5}}));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("x");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{2, 2}, {2, 2}, {3, 5}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(inplace_update, inplace_update_infershape_diff_test_4) {
  ge::op::InplaceUpdate op;
  op.UpdateInputDesc("x", create_desc_shape_range({2, 2, -1}, ge::DT_INT8,
                                                  ge::FORMAT_ND, {2, 2, -1},
                                                  ge::FORMAT_ND, {{2, 2}, {2, 2}, {2, 20}}));
  op.UpdateInputDesc("v", create_desc_shape_range({2, 2, -1}, ge::DT_INT32,
                                                  ge::FORMAT_ND, {2, 2, -1},
                                                  ge::FORMAT_ND, {{2, 2}, {2, 2}, {2, 20}}));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
