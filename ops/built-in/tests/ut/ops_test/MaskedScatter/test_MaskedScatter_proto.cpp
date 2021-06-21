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
 * @file test_MaskedScatter_proto.cpp
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
class masked_scatter:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"masked_scatter Proto Test SetUp"<<std::endl;
        }

        static void TearDownTestCase(){
            std::cout<<"masked_scatter Proto Test TearDown"<<std::endl;
        }
};


TEST_F(masked_scatter, masked_scatter_infershape_test_1){
    ge::op::MaskedScatter op;
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

    ge::TensorDesc tensor_desc_updates;
    ge::Shape updates_shape({X, Y});
    tensor_desc_updates.SetDataType(ge::DT_FLOAT);
    tensor_desc_updates.SetShape(updates_shape);
    
    op.UpdateInputDesc("x", tensor_desc_x);
    op.UpdateInputDesc("mask", tensor_desc_mask);
    op.UpdateInputDesc("updates", tensor_desc_updates);
 
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {X, Y};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);
}


TEST_F(masked_scatter, masked_scatter_infershape_test_2){
    ge::op::MaskedScatter op;
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

    ge::TensorDesc tensor_desc_updates;
    ge::Shape updates_shape({X, Y});
    tensor_desc_updates.SetDataType(ge::DT_FLOAT);
    tensor_desc_updates.SetShape(updates_shape);
    
    op.UpdateInputDesc("x", tensor_desc_x);
    op.UpdateInputDesc("mask", tensor_desc_mask);
    op.UpdateInputDesc("updates", tensor_desc_updates);
 
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {X, Y};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);
}
