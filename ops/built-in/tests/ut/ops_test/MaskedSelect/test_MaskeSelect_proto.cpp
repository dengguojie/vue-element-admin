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
 * @file test_MaskedSelect_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"
#include <string>
#include <vector>
class masked_select:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"masked_select Proto Test SetUp"<<std::endl;
        }

        static void TearDownTestCase(){
            std::cout<<"masked_select Proto Test TearDown"<<std::endl;
        }
};


TEST_F(masked_select, masked_select_infershape_test_1){
    ge::op::MaskedSelect op;

    int X = 32;
    int Y = 944;

    ge::TensorDesc tensor_desc_x;
    ge::Shape x_shape({X, Y});
    tensor_desc_x.SetDataType(ge::DT_FLOAT);
    tensor_desc_x.SetShape(x_shape);
    
    ge::TensorDesc tensor_desc_mask;
    ge::Shape mask_shape({Y});
    tensor_desc_mask.SetDataType(ge::DT_BOOL);
    tensor_desc_mask.SetShape(mask_shape);

    op.UpdateInputDesc("x", tensor_desc_x);
    op.UpdateInputDesc("mask", tensor_desc_mask);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {-1};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);
    std::vector<std::pair<int64_t,int64_t>> output_shape_range;
    EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range),ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{1,-1}};
    EXPECT_EQ(output_shape_range,expected_shape_range);
}

TEST_F(masked_select, masked_select_infershape_test_2){
    ge::op::MaskedSelect op;

    int X = 32;
    int Y = 944;

    ge::TensorDesc tensor_desc_x;
    ge::Shape x_shape({X, Y});
    tensor_desc_x.SetDataType(ge::DT_FLOAT);
    tensor_desc_x.SetShape(x_shape);
    
    ge::TensorDesc tensor_desc_mask;
    ge::Shape mask_shape({X, Y});
    tensor_desc_mask.SetDataType(ge::DT_BOOL);
    tensor_desc_mask.SetShape(mask_shape);

    op.UpdateInputDesc("x", tensor_desc_x);
    op.UpdateInputDesc("mask", tensor_desc_mask);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {-1};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);
}
