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
 * @file test_elu_grad_v2_proto.cpp
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

class EluGradV2Test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "elu_grad_v2 test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "elu_grad_v2 test TearDown" << std::endl;
    }
};

TEST_F(EluGradV2Test, elu_grad_v2_test_case_1) {
    ge::op::EluGradV2 elu_grad_v2_op;
    ge::TensorDesc input_desc1;
    ge::Shape shape({100,100});
    input_desc1.SetDataType(ge::DT_FLOAT16);
    input_desc1.SetShape(shape);
    input_desc1.SetOriginShape(shape);
    ge::TensorDesc input_desc2;
    input_desc2.SetDataType(ge::DT_FLOAT16);
    input_desc2.SetShape(shape);
    input_desc2.SetOriginShape(shape);

    elu_grad_v2_op.UpdateInputDesc("grads", input_desc1);
    elu_grad_v2_op.UpdateInputDesc("activations", input_desc2);

    auto ret = elu_grad_v2_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = elu_grad_v2_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {100,100};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
