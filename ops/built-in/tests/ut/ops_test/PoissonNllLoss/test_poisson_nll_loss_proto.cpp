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
 * @file test_poisson_nll_loss_proto.cpp
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

class PoissonNllLossTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "poisson_nll_loss test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "poisson_nll_loss test TearDown" << std::endl;
    }
};

TEST_F(PoissonNllLossTest, poisson_nll_loss_test_case_1) {
    ge::op::PoissonNllLoss poisson_nll_loss_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({1});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    poisson_nll_loss_op.SetAttr("reduction","mean");
    poisson_nll_loss_op.UpdateInputDesc("input_x", tensorDesc);
    poisson_nll_loss_op.UpdateInputDesc("target", tensorDesc);

    auto ret = poisson_nll_loss_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = poisson_nll_loss_op.GetOutputDesc("loss");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(PoissonNllLossTest, poisson_nll_loss_test_case_2) {
    ge::op::PoissonNllLoss poisson_nll_loss_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({1,2,3});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    poisson_nll_loss_op.SetAttr("reduction","none");
    poisson_nll_loss_op.UpdateInputDesc("input_x", tensorDesc);
    poisson_nll_loss_op.UpdateInputDesc("target", tensorDesc);

    auto ret = poisson_nll_loss_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = poisson_nll_loss_op.GetOutputDesc("loss");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1,2,3};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
