/**
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_less_equal_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include <vector>
#include "nn_norm_ops.h"

class LpLossTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "lp_loss SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "lp_loss TearDown" << std::endl;
  }
};


TEST_F(LpLossTest, lp_loss_test_case1_dynamic) {
  ge::op::LpLoss lp_loss_op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2,2},{100,200},{4,8}};
  ge::TensorDesc tensorDesc;
  tensorDesc = create_desc_shape_range({-1, 100, 4},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);

  lp_loss_op.UpdateInputDesc("predict", tensorDesc);
  lp_loss_op.UpdateInputDesc("label", tensorDesc);
  lp_loss_op.SetAttr("p", 1);
  lp_loss_op.SetAttr("reduction", "sum");

  auto ret = lp_loss_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto ret_Verify = lp_loss_op.VerifyAllAttr(true);
  EXPECT_EQ(ret_Verify, ge::GRAPH_SUCCESS);

  auto output_desc = lp_loss_op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape ={};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  std::vector<std::pair<int64_t,int64_t>> expected_output_shape_range = {};
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(LpLossTest, lp_loss_test_case2_dynamic) {
  ge::op::LpLoss lp_loss_op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2,2},{100,200},{4,8}};
  ge::TensorDesc tensorDesc;
  tensorDesc = create_desc_shape_range({-1, 100, 4},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);

  lp_loss_op.UpdateInputDesc("predict", tensorDesc);
  lp_loss_op.UpdateInputDesc("label", tensorDesc);
  lp_loss_op.SetAttr("p", 1);
  lp_loss_op.SetAttr("reduction", "none");

  auto ret = lp_loss_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto ret_Verify = lp_loss_op.VerifyAllAttr(true);
  EXPECT_EQ(ret_Verify, ge::GRAPH_SUCCESS);

  auto output_desc = lp_loss_op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape={-1, 100, 4};

  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  std::vector<std::pair<int64_t,int64_t>> expected_output_shape_range = shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
  }

TEST_F(LpLossTest, lp_loss_test_case3_dynamic) {
 ge::op::LpLoss lp_loss_op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2,2},{100,200},{4,8}};
  ge::TensorDesc tensorDesc;
  tensorDesc = create_desc_shape_range({-1, 100, 4},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);

  lp_loss_op.UpdateInputDesc("predict", tensorDesc);
  lp_loss_op.UpdateInputDesc("label", tensorDesc);
  lp_loss_op.SetAttr("p", 1);
  lp_loss_op.SetAttr("reduction", "reduce");

  auto ret_Verify = lp_loss_op.VerifyAllAttr(true);
  EXPECT_EQ(ret_Verify, ge::GRAPH_SUCCESS);

  auto ret = lp_loss_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
