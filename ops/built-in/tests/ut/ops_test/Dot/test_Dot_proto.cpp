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
 * @file test_Dot_proto.cpp
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
#include "elewise_calculation_ops.h"

class DotTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dot test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "dot test TearDown" << std::endl;
    }
};

TEST_F(DotTest, dot_test_case_1) {
    ge::op::Dot dot_op;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({2});
    tensorDesc1.SetDataType(ge::DT_FLOAT16);
    tensorDesc1.SetShape(shape1);
    tensorDesc1.SetOriginShape(shape1);
    dot_op.UpdateInputDesc("input_x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({2});
    tensorDesc2.SetDataType(ge::DT_FLOAT16);
    tensorDesc2.SetShape(shape2);
    tensorDesc2.SetOriginShape(shape2);
    dot_op.UpdateInputDesc("input_y", tensorDesc2);

    auto ret = dot_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = dot_op.GetOutputDesc("output");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}


TEST_F(DotTest, dot_test_case_2) {
    ge::op::Dot dot_op;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({2, 3, 4});
    tensorDesc1.SetDataType(ge::DT_FLOAT16);
    tensorDesc1.SetShape(shape1);
    dot_op.UpdateInputDesc("input_x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({3, 3, 4});
    tensorDesc2.SetDataType(ge::DT_FLOAT16);
    tensorDesc2.SetShape(shape2);
    dot_op.UpdateInputDesc("input_y", tensorDesc2);

    auto status = dot_op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = dot_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
