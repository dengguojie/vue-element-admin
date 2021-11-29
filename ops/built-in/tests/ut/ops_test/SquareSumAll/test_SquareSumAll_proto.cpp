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

class SquareSumAll : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "square_sum_v1_proto Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "square_sum_v1_proto Proto Test TearDown" << std::endl;
  }
};

TEST_F(SquareSumAll, square_sum_v1_infershape_test_01) {
  ge::op::SquareSumAll op;

  op.UpdateInputDesc("x1", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x2", create_desc({4, 3, 1}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc_y1 = op.GetOutputDesc("y1");
  EXPECT_EQ(output_desc_y1.GetDataType(), ge::DT_FLOAT16);
  auto output_desc_y2 = op.GetOutputDesc("y2");
  EXPECT_EQ(output_desc_y2.GetDataType(), ge::DT_FLOAT16);
}