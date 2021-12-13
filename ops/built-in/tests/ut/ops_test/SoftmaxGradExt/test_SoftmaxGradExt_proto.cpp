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
#include "nn_norm_ops.h"

class SoftmaxGradExtTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "softmax_grad_ext test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "softmax_grad_ext test TearDown" << std::endl;
    }
};

TEST_F(SoftmaxGradExtTest, softmax_grad_ext_test_case_1) {
    ge::op::SoftmaxGradExt softmax_grad_ext_op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({10, 20, 30});
    tensor_desc.SetDataType(ge::DT_FLOAT16);
    tensor_desc.SetShape(shape);
    tensor_desc.SetOriginShape(shape);

    softmax_grad_ext_op.UpdateInputDesc("grad", tensor_desc);
    softmax_grad_ext_op.UpdateInputDesc("x1", tensor_desc);
    softmax_grad_ext_op.UpdateInputDesc("x2", tensor_desc);

    auto ret = softmax_grad_ext_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = softmax_grad_ext_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {10, 20, 30};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}