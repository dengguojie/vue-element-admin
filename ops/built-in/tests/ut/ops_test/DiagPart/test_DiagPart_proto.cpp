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
 * @file test_DiagPart_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>  // NOLINT
#include <iostream>
#include "op_proto_test_util.h"  // NOLINT
#include "matrix_calculation_ops.h"  // NOLINT

class DiagPart : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DiagPart SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DiagPart TearDown" << std::endl;
  }
};

TEST_F(DiagPart, diag_part_infer_shape_fp16) {
  ge::op::DiagPart op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{4, 4096}};
  auto tensor_desc = create_desc_shape_range({-1, 32, -1, 32},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {32},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, 32};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {
    {4, 4096},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
