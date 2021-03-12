/*
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

class prelu:public testing::Test {
    protected:
        static void SetUpTestCase() {
            std::cout << "prelu Proto Test SetUp" << std::endl;
        }

        static void TearDownTestCase() {
            std::cout << "prelu Proto Test TearDown" << std::endl;
        }
};


TEST_F(prelu, prelu_infershape_diff_test) {
    ge::op::PRelu op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}};
    auto tensor_desc = create_desc_shape_range({-1},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {64},
                                                ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", tensor_desc);
    op.UpdateInputDesc("weight", tensor_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_y_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {-1};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_y_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{2, 100}};
    EXPECT_EQ(output_shape_range, expected_shape_range);
}

