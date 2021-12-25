/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this
 file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_poisson_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "matrix_calculation_ops.h"
#include "random_ops.h"
#include "op_proto_test_util.h"

class Poisson : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "Poisson SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "Poisson TearDown" << std::endl; }
};

TEST_F(Poisson, poisson_infer_shape_fp16) {
  ge::op::Poisson op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc_shape_x = create_desc_shape_range({2,2},
                                                  ge::DT_FLOAT16, ge::FORMAT_ND,
                                                  {2,2},
                                                  ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc_shape_x);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2,2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}