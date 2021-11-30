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
 * @file test_Lerp_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class LerpTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "Lerp test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "Lerp test TearDown" << std::endl;
    }
};

TEST_F(LerpTest, lerp_test_case_1) {
    ge::op::Lerp lerp_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 1, 2});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({1, 1, 2});
    tensorDesc2.SetDataType(ge::DT_FLOAT16);
    tensorDesc2.SetShape(shape2);
    tensorDesc2.SetOriginShape(shape2);

    ge::TensorDesc tensorDesc3;
    tensorDesc3.SetDataType(ge::DT_FLOAT16);
    tensorDesc3.SetShape(shape);
    tensorDesc3.SetOriginShape(shape);

    lerp_op.UpdateInputDesc("start", tensorDesc);
    lerp_op.UpdateInputDesc("end", tensorDesc2);
    lerp_op.UpdateInputDesc("weight", tensorDesc3);

    auto status = lerp_op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = lerp_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = lerp_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 1, 2};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(LerpTest, lerp_test_case_2) {
    ge::op::Lerp lerp_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 1, 2});
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({1, 1, 2});
    tensorDesc2.SetDataType(ge::DT_FLOAT16);
    tensorDesc2.SetShape(shape2);
    tensorDesc2.SetOriginShape(shape2);

    ge::TensorDesc tensorDesc3;
    tensorDesc3.SetDataType(ge::DT_FLOAT16);
    tensorDesc3.SetShape(shape);
    tensorDesc3.SetOriginShape(shape);

    lerp_op.UpdateInputDesc("start", tensorDesc);
    lerp_op.UpdateInputDesc("end", tensorDesc2);
    lerp_op.UpdateInputDesc("weight", tensorDesc3);

    auto status = lerp_op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(LerpTest, lerp_test_case_3) {
    ge::op::Lerp lerp_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 1, 2});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({1, 1, 2});
    tensorDesc2.SetDataType(ge::DT_FLOAT16);
    tensorDesc2.SetShape(shape2);
    tensorDesc2.SetOriginShape(shape2);

    ge::TensorDesc tensorDesc3;
    tensorDesc3.SetDataType(ge::DT_FLOAT);
    tensorDesc3.SetShape(shape);
    tensorDesc3.SetOriginShape(shape);

    lerp_op.UpdateInputDesc("start", tensorDesc);
    lerp_op.UpdateInputDesc("end", tensorDesc2);
    lerp_op.UpdateInputDesc("weight", tensorDesc3);

    auto status = lerp_op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}