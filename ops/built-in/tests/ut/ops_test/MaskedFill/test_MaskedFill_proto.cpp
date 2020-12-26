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
 * @file test_MaskedFill_proto.cpp
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
#include "selection_ops.h"

class MaskedFillTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "masked_fill test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "masked_fill test TearDown" << std::endl;
    }
};

TEST_F(MaskedFillTest, masked_fill_test_case_1) {
    ge::op::MaskedFill masked_fill_op;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({2, 3, 4});
    tensorDesc1.SetDataType(ge::DT_FLOAT16);
    tensorDesc1.SetShape(shape1);
    masked_fill_op.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({2, 3, 4});
    tensorDesc2.SetDataType(ge::DT_BOOL);
    tensorDesc2.SetShape(shape2);
    masked_fill_op.UpdateInputDesc("mask", tensorDesc2);

    ge::TensorDesc tensorDesc3;
    ge::Shape shape3({1});
    tensorDesc3.SetDataType(ge::DT_FLOAT16);
    tensorDesc3.SetShape(shape3);
    masked_fill_op.UpdateInputDesc("value", tensorDesc3);

    auto ret = masked_fill_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = masked_fill_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

