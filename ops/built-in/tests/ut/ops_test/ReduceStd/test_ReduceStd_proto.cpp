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
 * @file test_ReduceStd_proto.cpp
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
#include "reduce_ops.h"

class ReduceStdTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "reduce_std test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "reduce_std test TearDown" << std::endl;
    }
};

TEST_F(ReduceStdTest, reduce_std_test_case_1) {
    ge::op::ReduceStd reduce_std_op;

    ge::TensorDesc tensor_desc;
    ge::Shape shape1({3, 4, 5});
    tensor_desc.SetDataType(ge::DT_FLOAT16);
    tensor_desc.SetShape(shape1);
    reduce_std_op.UpdateInputDesc("x", tensor_desc);
    std::vector<int64_t> dim = {1};
    reduce_std_op.SetAttr("dim", dim);
    bool unbiased = true;
    reduce_std_op.SetAttr("unbiased", unbiased);
    bool keepdim = false;
    reduce_std_op.SetAttr("keepdim", keepdim);
    auto ret = reduce_std_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc1 = reduce_std_op.GetOutputDescByName("y1");
    EXPECT_EQ(output_desc1.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape1 = {3, 5};
    EXPECT_EQ(output_desc1.GetShape().GetDims(), expected_output_shape1);

    auto output_desc2 = reduce_std_op.GetOutputDescByName("y2");
    EXPECT_EQ(output_desc2.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape2 = {3, 5};
    EXPECT_EQ(output_desc2.GetShape().GetDims(), expected_output_shape2);
}
