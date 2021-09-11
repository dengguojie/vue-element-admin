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
 * @file test_soft_margin_loss_grad_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "math_ops.h"

class SoftMarginLossGradTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "soft_margin_loss_grad test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "soft_margin_loss_grad test TearDown" << std::endl;
    }
};

TEST_F(SoftMarginLossGradTest, soft_margin_loss_grad_test_case_1) {
    ge::op::SoftMarginLossGrad soft_margin_loss_grad_op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({100});
    tensor_desc.SetDataType(ge::DT_FLOAT16);
    tensor_desc.SetShape(shape);
    tensor_desc.SetOriginShape(shape);

    soft_margin_loss_grad_op.UpdateInputDesc("predict", tensor_desc);
    soft_margin_loss_grad_op.UpdateInputDesc("label", tensor_desc);
    soft_margin_loss_grad_op.UpdateInputDesc("dout", tensor_desc);

    auto ret = soft_margin_loss_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = soft_margin_loss_grad_op.GetOutputDesc("gradient");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {100};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SoftMarginLossGradTest, soft_margin_loss_grad_test_case_2) {
    ge::op::SoftMarginLossGrad soft_margin_loss_grad_op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({10, 20});
    tensor_desc.SetDataType(ge::DT_FLOAT16);
    tensor_desc.SetShape(shape);
    tensor_desc.SetOriginShape(shape);

    soft_margin_loss_grad_op.UpdateInputDesc("predict", tensor_desc);
    soft_margin_loss_grad_op.UpdateInputDesc("label", tensor_desc);
    soft_margin_loss_grad_op.UpdateInputDesc("dout", tensor_desc);

    auto ret = soft_margin_loss_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = soft_margin_loss_grad_op.GetOutputDesc("gradient");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {10, 20};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SoftMarginLossGradTest, soft_margin_loss_grad_test_case_3) {
    ge::op::SoftMarginLossGrad soft_margin_loss_grad_op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({10, 20, 30});
    tensor_desc.SetDataType(ge::DT_FLOAT);
    tensor_desc.SetShape(shape);
    tensor_desc.SetOriginShape(shape);

    soft_margin_loss_grad_op.UpdateInputDesc("predict", tensor_desc);
    soft_margin_loss_grad_op.UpdateInputDesc("label", tensor_desc);
    soft_margin_loss_grad_op.UpdateInputDesc("dout", tensor_desc);

    auto ret = soft_margin_loss_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = soft_margin_loss_grad_op.GetOutputDesc("gradient");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {10, 20, 30};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SoftMarginLossGradTest, soft_margin_loss_grad_test_case_4) {
    ge::op::SoftMarginLossGrad soft_margin_loss_grad_op;
    ge::TensorDesc predict_desc;
    ge::TensorDesc label_desc;
    ge::Shape predcit_shape({10, 20, 30});
    ge::Shape label_shape({10, 20, 1});

    predict_desc.SetDataType(ge::DT_FLOAT16);
    predict_desc.SetShape(predcit_shape);
    predict_desc.SetOriginShape(predcit_shape);
    label_desc.SetDataType(ge::DT_FLOAT16);
    label_desc.SetShape(label_shape);
    label_desc.SetOriginShape(label_shape);

    soft_margin_loss_grad_op.UpdateInputDesc("predict", predict_desc);
    soft_margin_loss_grad_op.UpdateInputDesc("label", label_desc);
    soft_margin_loss_grad_op.UpdateInputDesc("dout", predict_desc);

    auto ret = soft_margin_loss_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = soft_margin_loss_grad_op.GetOutputDesc("gradient");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {10, 20, 30};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}