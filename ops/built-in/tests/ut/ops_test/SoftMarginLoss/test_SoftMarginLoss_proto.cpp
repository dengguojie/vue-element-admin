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
 * @file test_SoftMarginLoss_proto.cpp
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
#include "nn_norm_ops.h"

class SoftMarginLossTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "soft_margin_loss test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "soft_margin_loss test TearDown" << std::endl;
    }
};

TEST_F(SoftMarginLossTest, soft_margin_loss_test_case_1) {
    ge::op::SoftMarginLoss op;

    ge::TensorDesc tensor_desc1;
    ge::Shape shape1({2});
    tensor_desc1.SetDataType(ge::DT_FLOAT16);
    tensor_desc1.SetShape(shape1);
    tensor_desc1.SetOriginShape(shape1);
    op.UpdateInputDesc("input_x", tensor_desc1);

    ge::TensorDesc tensor_desc2;
    ge::Shape shape2({2});
    tensor_desc2.SetDataType(ge::DT_FLOAT16);
    tensor_desc2.SetShape(shape2);
    tensor_desc2.SetOriginShape(shape2);
    op.UpdateInputDesc("input_y", tensor_desc2);

    std::string attr_value = "none";
    op.SetAttr("reduction", attr_value);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("output_z");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
