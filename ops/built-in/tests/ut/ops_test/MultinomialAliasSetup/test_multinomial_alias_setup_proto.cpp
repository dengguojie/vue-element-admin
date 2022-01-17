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
 * @file test_multinomial_alias_setup_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "random_ops.h"

class multinomial_alias_setup : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "multinomial_alias_setup SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "multinomial_alias_setup TearDown" << std::endl;
  }
};

TEST_F(multinomial_alias_setup, multinomial_alias_setup_infer_shape_fp32) {
  ge::op::MultinomialAliasSetup op;
  ge::TensorDesc tensor_desc;
  ge::Shape shape({100});
  tensor_desc.SetDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(shape);
  tensor_desc.SetOriginShape(shape);
  op.UpdateInputDesc("probs", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_q_desc = op.GetOutputDesc("q");
  auto output_j_desc = op.GetOutputDesc("j");
  std::vector<int64_t> expected_output_shape = {100};
  EXPECT_EQ(output_q_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_j_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_q_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_j_desc.GetDataType(), ge::DT_INT64);
}

TEST_F(multinomial_alias_setup, multinomial_alias_setup_infer_shape_fp64) {
  ge::op::MultinomialAliasSetup op;
  ge::TensorDesc tensor_desc;
  ge::Shape shape({200});
  tensor_desc.SetDataType(ge::DT_DOUBLE);
  tensor_desc.SetShape(shape);
  tensor_desc.SetOriginShape(shape);
  op.UpdateInputDesc("probs", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_q_desc = op.GetOutputDesc("q");
  auto output_j_desc = op.GetOutputDesc("j");
  std::vector<int64_t> expected_output_shape = {200};
  EXPECT_EQ(output_q_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_j_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_q_desc.GetDataType(), ge::DT_DOUBLE);
  EXPECT_EQ(output_j_desc.GetDataType(), ge::DT_INT64);
}
