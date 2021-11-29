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
 * @file test_acoshgrad_proto.cpp
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

class acoshgrad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "acoshgrad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "acoshgrad TearDown" << std::endl;
  }
};

TEST_F(acoshgrad, acoshgrad_infer_shape_fp16) {
  ge::op::AcoshGrad op;
  
  op.UpdateInputDesc("y", create_desc({-1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("dy", create_desc({-1}, ge::DT_FLOAT16));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("z");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(acoshgrad, acoshgrad_infer_same_test) {
  ge::op::AcoshGrad op;

  op.UpdateInputDesc("y", create_desc({1, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("dy", create_desc({1, 3, 4}, ge::DT_FLOAT16));
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("z");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}




