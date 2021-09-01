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

class LpNormReduceTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "LpNormReduce SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LpNormReduce TearDown" << std::endl;
  }
};

TEST_F(LpNormReduceTest, lp_norm_reduce_infer_shape_p_1_keepdim_true) {
  ge::op::LpNormReduce op;
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
  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 1, 1, 224};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(LpNormReduceTest, lp_norm_reduce_infer_shape_p_2_keepdim_false) {
  ge::op::LpNormReduce op;
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
  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {64, 224, 224};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(LpNormReduceTest, lp_norm_reduce_infer_shape_fp16) {
  ge::op::LpNormReduce op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{15, 16}, {8,8}, {375,375}};
    auto tensor_desc_x = create_desc_shape_range({-1,8,375},
                                                  ge::DT_FLOAT16, ge::FORMAT_ND,
                                                  {16,8,375},
                                                  ge::FORMAT_ND, shape_range);


  op.UpdateInputDesc("x", tensor_desc_x);
  op.SetAttr("p", 2);
  op.SetAttr("axes", {});
  op.SetAttr("keepdim", false);
  op.SetAttr("epsilon", 0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
