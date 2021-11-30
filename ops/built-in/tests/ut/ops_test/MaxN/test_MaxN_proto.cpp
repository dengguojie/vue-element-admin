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

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class MaxN_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaxN_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxN_UT TearDown" << std::endl;
  }
};

TEST_F(MaxN_UT, InferShapeMaxN_000) {
  ge::op::MaxN op;
  auto tensor_desc_1 =
      create_desc_with_ori({2, 100, 4}, ge::DT_INT32, ge::FORMAT_ND, {2, 100, 4}, ge::FORMAT_ND);
  auto tensor_desc_2 =
      create_desc_with_ori({2, 100, 4}, ge::DT_INT16, ge::FORMAT_ND, {2, 100, 4}, ge::FORMAT_ND);
  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc_1);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc_2);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxN_UT, InferShapeMaxN_001) {
  ge::op::MaxN op;
  op.create_dynamic_input_x(0);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxN_UT, InferShapeMaxN_002) {
  ge::op::MaxN op;
  auto tensor_desc =
      create_desc_with_ori({2, 100, 4}, ge::DT_INT32, ge::FORMAT_NCHW, {2, 100, 4}, ge::FORMAT_NCHW);
  op.create_dynamic_input_x(1);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(output_desc.GetFormat(), ge::FORMAT_NCHW);
  std::vector<int64_t> expected_output_shape = {2, 100, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}