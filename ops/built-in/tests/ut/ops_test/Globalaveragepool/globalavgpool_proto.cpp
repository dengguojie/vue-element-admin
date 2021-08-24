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
 * @file test_GeluGrad_proto.cpp
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"

class GlobalAveragePooltest : public testing::Test {
  protected:
    static void SetUpTestCase() {
      std::cout << "globalavgpool SetUp" << std::endl;
    }
    static void TearDownTestCase() {
      std::cout << "globalavgpool TearDown" << std::endl;
    }
};

//2d success
TEST_F(GlobalAveragePooltest, globalavgpool_case1) {
  ge::op::GlobalAveragePool op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 3, 5, 5}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 3, 5, 5}, ge::FORMAT_NCHW));
  op.UpdateOutputDesc("y", create_desc_with_ori({1, 3, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 3, 1, 1}, ge::FORMAT_NCHW));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

//3d success
TEST_F(GlobalAveragePooltest, globalavgpool_case2) {
  ge::op::GlobalAveragePool op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 5, 5, 4, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 5, 5, 4, 3}, ge::FORMAT_NCDHW));
  op.UpdateOutputDesc("y", create_desc_with_ori({1, 5, 1, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 5, 1, 1, 1}, ge::FORMAT_NCDHW));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

//1d success
TEST_F(GlobalAveragePooltest, globalavgpool_case3) {
  ge::op::GlobalAveragePool op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 5, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 5, 5}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y", create_desc_with_ori({1, 5, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 5, 1}, ge::FORMAT_ND));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}


//x_format error
TEST_F(GlobalAveragePooltest, globalavgpool_case4) {
  ge::op::GlobalAveragePool op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 3, 5, 5}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 3, 5, 5}, ge::FORMAT_NHWC));
  op.UpdateOutputDesc("y", create_desc_with_ori({1, 3, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 3, 1, 1}, ge::FORMAT_NHWC));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}