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
 * @file test_LayerNorm_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class LayerNormTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "LayerNorm SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LayerNorm TearDown" << std::endl;
  }
};

TEST_F(LayerNormTest, layer_norm_test_1) {
  ge::op::LayerNorm op;

  op.UpdateInputDesc("x", create_desc({30, 256, 512}, ge::DT_FLOAT));

  int begin_norm_axis = -4;
  op.SetAttr("begin_norm_axis", begin_norm_axis);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, 0);
}

TEST_F(LayerNormTest, layer_norm_test_2) {
  ge::op::LayerNorm op;

  op.UpdateInputDesc("x", create_desc({30, 256, 512}, ge::DT_FLOAT));
  op.UpdateInputDesc("mean", create_desc({512}, ge::DT_FLOAT));
  op.UpdateInputDesc("variance", create_desc({512}, ge::DT_FLOAT));

  int begin_norm_axis = -1;
  op.SetAttr("begin_norm_axis", begin_norm_axis);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y_desc = op.GetOutputDesc("y");
  auto output_mean_desc = op.GetOutputDesc("mean");
  auto output_var_desc = op.GetOutputDesc("variance");

  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_mean_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_var_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_y_shape = {30, 256, 512};
  std::vector<int64_t> expected_mv_shape = {30, 256, 1};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
  EXPECT_EQ(output_mean_desc.GetShape().GetDims(), expected_mv_shape);
  EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_mv_shape);
}

