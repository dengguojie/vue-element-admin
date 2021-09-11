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
 * @file test_Cummax_proto.cpp
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

class CummaxTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "cummax test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "cummax test TearDown" << std::endl;
    }
};

TEST_F(CummaxTest, cummax_test_case_1) {
    ge::op::Cummax cummax_op;

    ge::TensorDesc tensor_desc;
    ge::Shape shape1({32});
    tensor_desc.SetDataType(ge::DT_INT64);
    tensor_desc.SetShape(shape1);
    tensor_desc.SetOriginShape(shape1);
    cummax_op.UpdateInputDesc("x", tensor_desc);

    auto ret = cummax_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc1 = cummax_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc1.GetDataType(), ge::DT_INT64);
    std::vector<int64_t> expected_output_shape1 = {32};
    EXPECT_EQ(output_desc1.GetShape().GetDims(), expected_output_shape1);

    auto output_desc2 = cummax_op.GetOutputDesc("indices");
    std::vector<int64_t> expected_output_shape2 = {32};
    EXPECT_EQ(output_desc2.GetShape().GetDims(), expected_output_shape2);
}
