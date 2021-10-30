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
#include <climits>
#include "op_proto_test_util.h"
#include "util.h"
#include "graph/utils/op_desc_utils.h"
#include "array_ops.h"
#include "graph/ge_tensor.h"

using std::make_pair;
class GET_SHAPE_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GET_SHAPE_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GET_SHAPE_UT TearDown" << std::endl;
  }
};

TEST_F(GET_SHAPE_UT, InferShapeSizeRange_000) {
  ge::op::GetShape op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc =
      create_desc_shape_range({2, 100, 4}, ge::DT_INT32, ge::FORMAT_ND, {2, 100, 4}, ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);

  std::vector<int64_t> expected_output_shape = {9};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(GET_SHAPE_UT, InferShapeSizeRange_001) {
  ge::op::GetShape op;
  op.create_dynamic_input_x(1);
  op.UpdateDynamicInputDesc("x", 0, create_desc({-2}, ge::DT_INT32));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}