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

class TensorEqual_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TensorEqual_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TensorEqual_UT TearDown" << std::endl;
  }
};

TEST_F(TensorEqual_UT, InferShapeTensorEqual_000) {
  ge::op::TensorEqual op;
  auto tensor_desc_1 =
      create_desc_with_ori({2, 100, 4}, ge::DT_INT32, ge::FORMAT_ND, {2, 100, 4}, ge::FORMAT_ND);
  auto tensor_desc_2 =
      create_desc_with_ori({2, 100, 4}, ge::DT_INT16, ge::FORMAT_ND, {2, 100, 4}, ge::FORMAT_ND);
  op.UpdateInputDesc("input_x", tensor_desc_1);
  op.UpdateInputDesc("input_y", tensor_desc_2);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(TensorEqual_UT, InferShapeTensorEqual_001) {
  ge::op::TensorEqual op;
  auto tensor_desc_1 =
      create_desc_with_ori({2, 100, 4}, ge::DT_INT32, ge::FORMAT_ND, {2, 100, 4}, ge::FORMAT_ND);
  auto tensor_desc_2 =
      create_desc_with_ori({2, 100}, ge::DT_INT32, ge::FORMAT_ND, {2, 100}, ge::FORMAT_ND);
  op.UpdateInputDesc("input_x", tensor_desc_1);
  op.UpdateInputDesc("input_y", tensor_desc_2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(TensorEqual_UT, InferShapeTensorEqual_002) {
  ge::op::TensorEqual op;
  auto tensor_desc =
      create_desc_with_ori({2, 100, 4}, ge::DT_INT32, ge::FORMAT_NCHW, {2, 100, 4}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("input_x", tensor_desc);
  op.UpdateInputDesc("input_y", tensor_desc);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}