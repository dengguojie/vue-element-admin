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

class SquareSumV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "square_sum_v2_proto Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "square_sum_v2_proto Proto Test TearDown" << std::endl;
  }
};

TEST_F(SquareSumV2, InfershapeSquareSumV2_001) {
  ge::op::SquareSumV2 op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_FLOAT16));
  op.SetAttr("axis", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SquareSumV2, InfershapeSquareSumV2_002) {
  ge::op::SquareSumV2 op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_FLOAT16));
  std::vector<int64_t> axis = {2, 4};
  op.SetAttr("axis", axis);
  op.SetAttr("keep_dims", axis);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SquareSumV2, InfershapeSquareSumV2_003) {
  ge::op::SquareSumV2 op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_FLOAT16));
  op.UpdateInputDesc("input_x", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  std::vector<int64_t> axis = {};
  op.SetAttr("axis", axis);
  op.SetAttr("keep_dims", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SquareSumV2, InfershapeSquareSumV2_004) {
  ge::op::SquareSumV2 op;
  op.UpdateInputDesc("x", create_desc({2}, ge::DT_FLOAT16));
  op.UpdateInputDesc("input_x", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  std::vector<int64_t> axis = {-1, 2, 4};
  op.SetAttr("axis", axis);
  op.SetAttr("keep_dims", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}