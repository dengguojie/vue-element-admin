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
 * @file test_truncate_div_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class Bias : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "bias SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "bias TearDown" << std::endl;
  }
};

TEST_F(Bias, bias_infershape_test) {
  ge::op::Bias op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 10},{3,10},{4,10}};
  auto tensor_desc_x = create_desc_shape_range({-1,-1,-1},
                                               ge::DT_FLOAT16, ge::FORMAT_ND,
                                               {2,3,4},
                                               ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("axis", 0);
  op.SetAttr("num_axes", 3);
  op.SetAttr("bias_from_blob", true);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1,-1,-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
    {2, 10},{3,10},{4,10}
  };
  
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
