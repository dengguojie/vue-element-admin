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
 * @file test_hard_sigmoid_loss_grad_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

class HardSigmoidGradTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "hard_sigmoid_grad test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "hard_sigmoid_grad test TearDown" << std::endl;
    }
};

TEST_F(HardSigmoidGradTest, hard_sigmoid_grad_test_case_1) {
     ge::op::HardSigmoidGrad hard_sigmoid_grad_op;
     ge::TensorDesc tensor_desc;
     ge::Shape shape({100});
     tensor_desc.SetDataType(ge::DT_FLOAT16);
     tensor_desc.SetShape(shape);
     tensor_desc.SetOriginShape(shape);

     hard_sigmoid_grad_op.UpdateInputDesc("grads", tensor_desc);
     hard_sigmoid_grad_op.UpdateInputDesc("input_x", tensor_desc);

     auto ret = hard_sigmoid_grad_op.InferShapeAndType();
     EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

     auto output_desc = hard_sigmoid_grad_op.GetOutputDesc("y");
     EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
     std::vector<int64_t> expected_output_shape = {100};
     EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
