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
 * @file test_L1LossGrad_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class L1LossGradTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "L1LossGrad SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "L1LossGrad TearDown" << std::endl;
    }
};

TEST_F(L1LossGradTest, l1lossgrad_infershape_test1) {
    ge::op::L1LossGrad op;
    op.UpdateInputDesc("predict", create_desc({-1, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("grads", create_desc({-1, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("label", create_desc({-1, 8}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {-1, 8};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(L1LossGradTest, l1lossgrad_infershape_test2) {
    ge::op::L1LossGrad op;
    op.UpdateInputDesc("predict", create_desc({-1, -1, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("grads", create_desc({-1, -1, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("label", create_desc({-1, -1, 8}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {-1, -1, 8};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(L1LossGradTest, l1lossgrad_infershape_test3) {
    ge::op::L1LossGrad op;
    op.UpdateInputDesc("predict", create_desc({4, 16, 36}, ge::DT_FLOAT));
    op.UpdateInputDesc("grads", create_desc({4, 16, 36}, ge::DT_FLOAT));
    op.UpdateInputDesc("label", create_desc({4, 16, 36}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {4, 16, 36};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(L1LossGradTest, l1lossgrad_infershape_test4) {
    ge::op::L1LossGrad op;
    op.UpdateInputDesc("predict", create_desc({4, 16, 36}, ge::DT_FLOAT));
    op.UpdateInputDesc("grads", create_desc({4, 16, 36}, ge::DT_FLOAT16));
    op.UpdateInputDesc("label", create_desc({4, 16, 36}, ge::DT_FLOAT16));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(L1LossGradTest, l1lossgrad_infershape_test5) {
    ge::op::L1LossGrad op;
    op.UpdateInputDesc("predict", create_desc({4, 16, 36}, ge::DT_FLOAT));
    op.UpdateInputDesc("grads", create_desc({4, 16, 36}, ge::DT_FLOAT16));
    op.UpdateInputDesc("label", create_desc({4, 16, 36}, ge::DT_INT32));

    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(L1LossGradTest, l1lossgrad_infershape_test6) {
    ge::op::L1LossGrad op;
    op.UpdateInputDesc("predict", create_desc({4, 16, 36}, ge::DT_FLOAT));
    op.UpdateInputDesc("grads", create_desc({4, 16}, ge::DT_FLOAT));
    op.UpdateInputDesc("label", create_desc({4}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(L1LossGradTest, l1lossgrad_infershape_test7) {
    ge::op::L1LossGrad op;
    op.UpdateInputDesc("predict", create_desc({-1, 16, 36}, ge::DT_FLOAT));
    op.UpdateInputDesc("grads", create_desc({-1, 16}, ge::DT_FLOAT));
    op.UpdateInputDesc("label", create_desc({-1}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}