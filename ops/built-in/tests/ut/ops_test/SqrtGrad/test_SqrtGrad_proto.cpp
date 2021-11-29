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

class SqrtGrad_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SqrtGrad_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SqrtGrad_UT TearDown" << std::endl;
  }
};

TEST_F(SqrtGrad_UT, InferShapeSqrtGrad_001) {
  ge::op::SqrtGrad op;
  op.UpdateInputDesc("y", create_desc_shape_range({2, 2, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 2, 1}, ge::FORMAT_ND,
                                                  {{2, 2}, {2, 2}, {1, 1}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("z");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 2, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{2, 2}, {2, 2}, {1, 1}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(SqrtGrad_UT, InferShapeSqrtGrad_002) {
  ge::op::SqrtGrad op;
  op.UpdateInputDesc("y", create_desc_shape_range({3, 4, 5, 6, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 4, 5, 6, -1},
                                                  ge::FORMAT_ND, {{3, 3}, {4, 4}, {5, 5}, {6, 6}, {3, 8}}));
  op.UpdateInputDesc("dy", create_desc_shape_range({3, 4, 5, 6, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {3, 4, 5, 6, -1},
                                                   ge::FORMAT_ND, {{3, 3}, {4, 4}, {5, 5}, {6, 6}, {3, 8}}));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}