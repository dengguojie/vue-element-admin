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
 * @file test_apply_add_sign_d_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"

class ApplyAddSignD : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyAddSignD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyAddSignD Proto Test TearDown" << std::endl;
  }
};

TEST_F(ApplyAddSignD, dynamic_apply_add_sign_d_test_01) {
  ge::op::ApplyAddSignD op;

  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 2}, {1, 7}, {1, 16}};

  auto tensor_desc = create_desc_shape_range({-1, -1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 7, 16},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("var", tensor_desc);
  op.UpdateInputDesc("m", tensor_desc);
  op.UpdateInputDesc("lr", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("alpha", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("sign_decay", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", tensor_desc);
  op.SetAttr("use_locking", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = shape_range;

  auto output_desc_var = op.GetOutputDesc("var");
  EXPECT_EQ(output_desc_var.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc_var.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc_var.GetOriginShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_var_shape_range;
  EXPECT_EQ(output_desc_var.GetShapeRange(output_var_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_var_shape_range, expected_shape_range);

  auto output_desc_m = op.GetOutputDesc("m");
  EXPECT_EQ(output_desc_m.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc_m.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc_m.GetOriginShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_m_shape_range;
  EXPECT_EQ(output_desc_m.GetShapeRange(output_m_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_m_shape_range, expected_shape_range);
}
