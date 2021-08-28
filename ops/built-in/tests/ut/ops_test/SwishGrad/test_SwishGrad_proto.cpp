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
 * @file test_SwishGrad_proto.cpp
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

class SwishGrad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SwishGrad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SwishGrad TearDown" << std::endl;
  }
};

TEST_F(SwishGrad, swish_grad_infershape_test){
  ge::op::SwishGrad op;
  
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 16},{1, 16}};

  auto tensor_desc = create_desc_shape_range({-1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {16, 16},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("grad", tensor_desc);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("y", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("grad_x");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{1,16},{1,16}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
