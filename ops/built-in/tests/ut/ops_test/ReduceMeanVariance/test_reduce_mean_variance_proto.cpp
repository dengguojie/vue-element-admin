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
 * @file test_reducemean_d_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class ReduceMeanVariance : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ReduceMeanVariance SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ReduceMeanVariance TearDown" << std::endl;
  }
};

TEST_F(ReduceMeanVariance, reduce_mean_variance_infer_shape_fp16) {
    ge::op::ReduceMeanVariance op;

    op.UpdateInputDesc("x", create_desc({4, 224, 224, 260, 32}, ge::DT_FLOAT16));
    std::vector<int64_t> axes = {1, 2, 3};
    op.SetAttr("axes", axes);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_mean_desc = op.GetOutputDesc("mean");
    auto output_var_desc = op.GetOutputDesc("variance");

    std::vector<int64_t> expected_y_shape = {4, 1, 1, 1, 32};
    std::vector<int64_t> expected_mv_shape = {4, 1, 1, 1, 32};
    EXPECT_EQ(output_mean_desc.GetShape().GetDims(), expected_mv_shape);
    EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_mv_shape);
}
