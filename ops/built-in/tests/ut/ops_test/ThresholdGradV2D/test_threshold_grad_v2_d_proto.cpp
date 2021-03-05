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
 * @file test_threshold_grad_v2_d_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>  // NOLINT
#include <iostream>
#include "op_proto_test_util.h"  // NOLINT
#include "nonlinear_fuc_ops.h"  // NOLINT

class threshold_grad_v2_d : public testing::Test {
 protected:
    static void SetUpTestCase() {
        std::cout << "threshold_grad_v2_d SetUp" << std::endl;
    }

    static void TearDownTestCase() {
    std::cout << "threshold_grad_v2_d Test TearDown" << std::endl;
  }
};

TEST_F(threshold_grad_v2_d, cce_threshold_grad_v2_d_diff_shape_success) {
    ge::op::ThresholdGradV2D op;
    op.UpdateInputDesc("gradients", create_desc({1, 2, 3, 5}, ge::DT_FLOAT16));
    op.UpdateInputDesc("features", create_desc({1, 2, 3, 1}, ge::DT_FLOAT16));
    op.SetAttr("threshold", 1.2f);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("backprops");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, 2, 3, 5};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(threshold_grad_v2_d, cce_threshold_grad_v2_d_float32) {
    ge::op::ThresholdGradV2D op;
    op.UpdateInputDesc("gradients", create_desc({1, 2, 3, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("features", create_desc({1, 2, 3, 5}, ge::DT_FLOAT));
    op.SetAttr("threshold", 1.2f);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("backprops");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {1, 2, 3, 5};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
TEST_F(threshold_grad_v2_d, cce_threshold_grad_v2_d_verify_fail) {
    ge::op::ThresholdGradV2D op;
    op.UpdateInputDesc("gradients", create_desc({1, 2, 3, 5}, ge::DT_FLOAT16));
    op.UpdateInputDesc("features", create_desc({1, 2, 3, 5}, ge::DT_FLOAT));
    op.SetAttr("threshold", 1.2f);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(threshold_grad_v2_d, cce_threshold_grad_v2_d_infer_shape_fail) {
    ge::op::ThresholdGradV2D op;
    op.UpdateInputDesc("gradients", create_desc({1, 2, 3, 5}, ge::DT_FLOAT16));
    op.UpdateInputDesc("features", create_desc({1, 2, 3, 2}, ge::DT_FLOAT16));
    op.SetAttr("threshold", 1.2f);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
