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
 * @file test_PtMuls_proto.cpp
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

class PtMulsTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "pt_muls test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "pt_muls test TearDown" << std::endl;
    }
};

TEST_F(PtMulsTest, pt_muls_test_case_1) {
    ge::op::PtMuls pt_muls_op;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({2, 3, 4});
    tensorDesc1.SetDataType(ge::DT_FLOAT16);
    tensorDesc1.SetShape(shape1);
    pt_muls_op.UpdateInputDesc("x1", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({1});
    tensorDesc2.SetDataType(ge::DT_FLOAT16);
    tensorDesc2.SetShape(shape2);
    pt_muls_op.UpdateInputDesc("x2", tensorDesc2);

    auto ret = pt_muls_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = pt_muls_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}


TEST_F(PtMulsTest, pt_muls_test_case_2) {
    ge::op::PtMuls pt_muls_op;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({2, 3, 4});
    tensorDesc1.SetDataType(ge::DT_FLOAT16);
    tensorDesc1.SetShape(shape1);
    pt_muls_op.UpdateInputDesc("x1", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({3, 3, 4});
    tensorDesc2.SetDataType(ge::DT_FLOAT16);
    tensorDesc2.SetShape(shape2);
    pt_muls_op.UpdateInputDesc("x2", tensorDesc2);

    auto ret = pt_muls_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
