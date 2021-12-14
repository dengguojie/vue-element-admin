/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_FusedMulAddN_proto.cpp
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class fused_mul_add_n : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "fused_mul_add_n SetUp" << std::endl;
  }

  static void TearDownTestCase() {
      std::cout << "fused_mul_add_n TearDown" << std::endl;
  }
};

TEST_F(fused_mul_add_n, fused_mul_add_n_case) {
  ge::op::FusedMulAddN op;

  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 16}, {1, 16}};

  auto tensor_desc = create_desc_shape_range({-1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {16, 16},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc);
  op.UpdateInputDesc("x3", create_desc({1, }, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1, 16}, {1, 16}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(fused_mul_add_n, VerifyFusedMulAddN_001) {
  ge::op::FusedMulAddN op;
  op.UpdateInputDesc("x3", create_desc({1}, ge::DT_INT64));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(fused_mul_add_n, VerifyFusedMulAddN_002) {
  ge::op::FusedMulAddN op;
  op.UpdateInputDesc("x1", create_desc({1}, ge::DT_INT64));
  op.UpdateInputDesc("x2", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("x3", create_desc({3}, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}