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
 * @file test_div_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

// ----------------Div-------------------
class div : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "div SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "div TearDown" << std::endl;
    }
};

TEST_F(div, div_infershape_test_0) {
ge::op::Div op;
op.UpdateInputDesc("x1", create_desc_shape_range({2, 2, 1}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 1}, ge::FORMAT_ND, {{2,2},{2,2},{1,1}}));
op.UpdateInputDesc("x2", create_desc_shape_range({2, 2, 3}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 3}, ge::FORMAT_ND, {{2,2},{2,2},{3,3}}));
auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
std::vector<int64_t> expected_output_shape = {2, 2, 3};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
std::vector<std::pair<int64_t, int64_t>> output_shape_range;
output_desc.GetShapeRange(output_shape_range);
EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(div, div_infershape_test_1) {
ge::op::Div op;
op.UpdateInputDesc("x1", create_desc_shape_range({3, 4, 5, 6, -1}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, -1}, ge::FORMAT_ND, {{3,3},{4,4},{5,5},{6,6},{3,8}}));
op.UpdateInputDesc("x2", create_desc_shape_range({3, 4, 5, 6, 1}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 1}, ge::FORMAT_ND, {{3,3},{4,4},{5,5},{6,6},{1,1}}));
auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
std::vector<int64_t> expected_output_shape = {3, 4, 5, 6, -1};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{3,3},{4,4},{5,5},{6,6},{3,8}};
std::vector<std::pair<int64_t, int64_t>> output_shape_range;
output_desc.GetShapeRange(output_shape_range);
EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(div, div_infershape_test_2) {
ge::op::Div op;
op.UpdateInputDesc("x1", create_desc_shape_range({-1,2}, ge::DT_INT32, ge::FORMAT_ND, {-1,2}, ge::FORMAT_ND, {{1,8}, {2,2}}));
op.UpdateInputDesc("x2", create_desc_shape_range({1}, ge::DT_INT32, ge::FORMAT_ND, {1}, ge::FORMAT_ND, {}));
auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
std::vector<int64_t> expected_output_shape = {-1,2};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,8}, {2,2}};
std::vector<std::pair<int64_t, int64_t>> output_shape_range;
output_desc.GetShapeRange(output_shape_range);
EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(div, div_infershape_test_3) {
ge::op::Div op;
op.UpdateInputDesc("x1", create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {-1}, ge::FORMAT_ND, {{1,55}}));
op.UpdateInputDesc("x2", create_desc_shape_range({17, 2, 5, 1}, ge::DT_INT32, ge::FORMAT_ND, {17, 2, 5, 1}, ge::FORMAT_ND, {{17,17},{2,2},{5,5},{1,1}}));
auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
std::vector<int64_t> expected_output_shape = {17, 2, 5, -1};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{17,17},{2,2},{5,5},{1,55}};
std::vector<std::pair<int64_t, int64_t>> output_shape_range;
output_desc.GetShapeRange(output_shape_range);
EXPECT_EQ(output_shape_range, expected_output_shape_range);
}
