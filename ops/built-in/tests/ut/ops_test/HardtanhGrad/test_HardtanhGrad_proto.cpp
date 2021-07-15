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
 * @file test_HardtanhGrad_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

class hardtanh_grad:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"hardtanh_grad Proto Test SetUp"<<std::endl;
        }

        static void TearDownTestCase(){
            std::cout<<"hardtanh_grad Proto Test TearDown"<<std::endl;
        }
};


TEST_F(hardtanh_grad, hardtanh_grad_infershape_diff_test){
    ge::op::HardtanhGrad op;
    auto tensor_desc = create_desc_shape_range({-1, 8, 375},
                                               ge::DT_FLOAT16, ge::FORMAT_ND,
                                               {16, 8, 375},
                                               ge::FORMAT_ND, {{16, 16}, {8, 8}, {375, 375}});

    op.UpdateInputDesc("result", tensor_desc);
    op.UpdateInputDesc("grad", tensor_desc);
    float min_val = -1.0;
    float max_val = 1.0;
    op.SetAttr("min_val", min_val);
    op.SetAttr("max_val", max_val);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {-1, 8, 375};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{16, 16}, {8, 8}, {375, 375}};
    EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(hardtanh_grad, hardtanh_grad_infershape_diff_test2){
    ge::op::HardtanhGrad op;
    auto tensor_desc1 = create_desc_shape_range({-1, 8, 35},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {16, 8, 35},
                                                ge::FORMAT_ND, {{16, 16}, {8, 8}, {35, 35}});
    auto tensor_desc2 = create_desc_shape_range({-1, 8, 35},
                                                ge::DT_FLOAT, ge::FORMAT_ND,
                                                {16, 8, 35},
                                                ge::FORMAT_ND, {{16, 16}, {8, 8}, {35, 35}});
    op.UpdateInputDesc("result", tensor_desc1);
    op.UpdateInputDesc("grad", tensor_desc2);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
