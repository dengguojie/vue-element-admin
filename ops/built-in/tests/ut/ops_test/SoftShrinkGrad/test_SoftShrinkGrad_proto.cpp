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
 * @file test_SoftShrinkGrad_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

class SoftShrinkGradTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "SoftShrink test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "SoftShrink test TearDown" << std::endl;
    }
};

TEST_F(SoftShrinkGradTest, soft_shrink_grad_test_case_1) {
    ge::op::SoftShrinkGrad soft_shrink_grad_op;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({2, 3, 4});
    tensorDesc1.SetDataType(ge::DT_FLOAT16);
    tensorDesc1.SetShape(shape1);
    tensorDesc1.SetOriginShape(shape1);
    soft_shrink_grad_op.UpdateInputDesc("input_x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({3, 4});
    tensorDesc2.SetDataType(ge::DT_FLOAT16);
    tensorDesc2.SetShape(shape2);
    soft_shrink_grad_op.UpdateInputDesc("input_grad", tensorDesc2);

    auto ret = soft_shrink_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = soft_shrink_grad_op.GetOutputDesc("output_y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SoftShrinkGradTest, soft_shrink_grad_test_case_2) {
    ge::op::SoftShrinkGrad soft_shrink_grad_op;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({2, 3, 4});
    tensorDesc1.SetDataType(ge::DT_FLOAT16);
    tensorDesc1.SetShape(shape1);
    tensorDesc1.SetOriginShape(shape1);
    soft_shrink_grad_op.UpdateInputDesc("input_x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({3, 3, 4});
    tensorDesc2.SetDataType(ge::DT_FLOAT16);
    tensorDesc2.SetShape(shape2);
    tensorDesc2.SetOriginShape(shape2);
    soft_shrink_grad_op.UpdateInputDesc("input_grad", tensorDesc2);

    auto ret = soft_shrink_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}