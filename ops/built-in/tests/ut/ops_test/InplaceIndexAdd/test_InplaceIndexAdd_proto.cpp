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
 * @file test_InplaceIndexAdd_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class inplace_index_add:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"inplace_index_add Proto Test SetUp"<<std::endl;
        }

        static void TearDownTestCase(){
            std::cout<<"inplace_index_add Proto Test TearDown"<<std::endl;
        }
};


TEST_F(inplace_index_add, inplace_index_add_infershape_diff_test){
    ge::op::InplaceIndexAdd op;
    auto tensor_desc_var = create_desc_shape_range({-1, 8, 375},
                                                   ge::DT_INT32, ge::FORMAT_ND,
                                                   {16, 8, 375},
                                                   ge::FORMAT_ND, {{16, 16}, {8, 8}, {375, 375}});
    auto tensor_desc_indices = create_desc_shape_range({-1},
                                                       ge::DT_INT32, ge::FORMAT_ND,
                                                       {3},
                                                       ge::FORMAT_ND, {{3, 3}});
    auto tensor_desc_updates = create_desc_shape_range({-1},
                                                       ge::DT_INT32, ge::FORMAT_ND,
                                                       {16, 3, 375},
                                                       ge::FORMAT_ND, {{16, 16}, {3, 3}, {375, 375}});
                                               

    op.UpdateInputDesc("var", tensor_desc_var);
    op.UpdateInputDesc("indices", tensor_desc_indices);
    op.UpdateInputDesc("updates", tensor_desc_updates);
    int64_t axis = 1;
    op.SetAttr("axis", axis);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    std::cout<<"inplace_index_add Proto 111111"<<std::endl;
    auto output_desc = op.GetOutputDesc("var");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
    std::vector<int64_t> expected_output_shape = {-1, 8, 375};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{16, 16}, {8, 8}, {375, 375}};
    EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(inplace_index_add, inplace_index_add_infershape_diff_test2){
    ge::op::InplaceIndexAdd op;
    auto tensor_desc_var = create_desc_shape_range({-1, 8, 375},
                                                   ge::DT_INT32, ge::FORMAT_ND,
                                                   {16, 8, 375},
                                                   ge::FORMAT_ND, {{16, 16}, {8, 8}, {375, 375}});
    auto tensor_desc_indices = create_desc_shape_range({-1},
                                                       ge::DT_INT32, ge::FORMAT_ND,
                                                       {3},
                                                       ge::FORMAT_ND, {{3, 3}});
    auto tensor_desc_updates = create_desc_shape_range({-1},
                                                       ge::DT_FLOAT, ge::FORMAT_ND,
                                                       {16, 3, 375},
                                                       ge::FORMAT_ND, {{16, 16}, {3, 3}, {375, 375}});
                                               

    op.UpdateInputDesc("var", tensor_desc_var);
    op.UpdateInputDesc("indices", tensor_desc_indices);
    op.UpdateInputDesc("updates", tensor_desc_updates);
    int64_t axis = 1;
    op.SetAttr("axis", axis);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
