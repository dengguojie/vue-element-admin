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
 * @file test_BatchNorm_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "math_ops.h"

class LpNorm : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "LpNorm SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LpNorm TearDown" << std::endl;
  }
};

TEST_F(LpNorm, lp_norm_infer_shape_p_1_keepdim_true) {
  ge::op::LpNorm op;
  auto tensor_desc_x = create_desc_shape_range({2, 64, 224, 224},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {2, 64, 224, 224},
                                                ge::FORMAT_ND, {{2, 2}, {64, 64}, {224, 224}, {224, 224}});
                                                                                                                                    
  op.UpdateInputDesc("x", tensor_desc_x);
  op.SetAttr("p", 1);
  op.SetAttr("axes", {1, 2}); 
  op.SetAttr("keepdim", true);
  op.SetAttr("epsilon", 0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 1, 1, 224};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(LpNorm, lp_norm_infer_shape_p_2_keepdim_false) {
  ge::op::LpNorm op;
  auto tensor_desc_x = create_desc_shape_range({1, 2, 64, 224, 224},
                                                ge::DT_FLOAT, ge::FORMAT_ND,
                                                {1, 2, 64, 224, 224},
                                                ge::FORMAT_ND, {{1, 1}, {2, 2}, {64, 64}, {224, 224}, {224, 224}});
                                                                                                                                    
  op.UpdateInputDesc("x", tensor_desc_x);
  op.SetAttr("p", 2);
  op.SetAttr("axes", {0, 1}); 
  op.SetAttr("keepdim", false);
  op.SetAttr("epsilon", 0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {64, 224, 224};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}