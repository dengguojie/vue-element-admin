/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "graph/utils/op_desc_utils.h"
#include "all_ops.h"
#include "common/utils/ut_op_common.h"

class Transpose : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Transpose SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Transpose TearDown" << std::endl;
  }
};

using namespace ut_util;

TEST_F(Transpose, Transpose_const_infer_int32) {
  using namespace ge;
  // input x info
  auto input_x_shape = vector<int64_t>({2, 3, 4, 5});
  auto input_x_dtype = DT_FLOAT;
  // input perm info
  auto input_perm_shape = vector<int64_t>({4});
  auto perm_dtype = DT_INT32;
  vector<int32_t> perm_value = {3, 2, 1, 0};
  // expect result info
  std::vector<int64_t> expected_output_shape = {5, 4, 3, 2};

  // gen Transpose op
  auto test_op = op::Transpose("Transpose");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, perm, input_perm_shape, perm_dtype, FORMAT_ND, perm_value);

  // cmp the result with runtime op
  vector<bool> input_const = {false, true};
  CommonInferShapeOperatorWithConst(test_op, input_const, {}, {expected_output_shape});

  // cmp the result with old op
  test_op.InferShapeAndType();
  auto output_desc = test_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetDataType(), input_x_dtype);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(Transpose, Transpose_const_infer_int64) {
  using namespace ge;
  // input x info
  auto input_x_shape = vector<int64_t>({2, 3, 4, 5});
  auto input_x_dtype = DT_FLOAT;
  // input perm info
  auto input_perm_shape = vector<int64_t>({4});
  auto perm_dtype = DT_INT64;
  vector<int64_t> perm_value = {0, 2, 1, 3};
  // expect result info
  std::vector<int64_t> expected_output_shape = {2, 4, 3, 5};

  // gen Transpose op
  auto test_op = op::Transpose("Transpose");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, perm, input_perm_shape, perm_dtype, FORMAT_ND, perm_value);

  // cmp the result with runtime op
  vector<bool> input_const = {false, true};
  CommonInferShapeOperatorWithConst(test_op, input_const, {}, {expected_output_shape});

  // cmp the result with old op
  test_op.InferShapeAndType();
  auto output_desc = test_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetDataType(), input_x_dtype);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
