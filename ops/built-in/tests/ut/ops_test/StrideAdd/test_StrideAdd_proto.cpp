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

class StrideAdd_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StrideAdd_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StrideAdd_UT TearDown" << std::endl;
  }
};

TEST_F(StrideAdd_UT, InferShapeStrideAdd_000) {
  ge::op::StrideAdd op;
  auto tensor_desc =
      create_desc_with_ori({2, 16, 4, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 16, 4, 4, 4}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc);
  op.SetAttr("c1_len", 4);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc.GetFormat(), ge::FORMAT_NCHW);
  std::vector<int64_t> expected_output_shape = {2, 4, 4, 4, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}