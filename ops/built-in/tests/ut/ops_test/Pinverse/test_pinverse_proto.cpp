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
 * @file test_Pinverse_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "matrix_calculation_ops.h"
#include "op_proto_test_util.h"

class pinverse : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "pinverse SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "pinverse TearDown" << std::endl; }
};

TEST_F(pinverse, Pinverse_infer_shape) {
  ge::op::Pinverse op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 3}};
  auto tensor_desc = create_desc_shape_range({2, 3},
                                             ge::DT_FLOAT, ge::FORMAT_ND, 
                                             {2, 3},
                                             ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {3, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
}
