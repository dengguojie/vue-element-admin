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
 * @file test_MseLossGrad_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class mse_loss_grad:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"mse_loss_grad Proto Test SetUp"<<std::endl;
        }

        static void TearDownTestCase(){
            std::cout<<"mse_loss_grad Proto Test TearDown"<<std::endl;
        }
};


TEST_F(mse_loss_grad, mse_loss_grad_infershape_diff_test){
    ge::op::MseLossGrad op;
    auto tensor_desc = create_desc_shape_range({-1,8,375},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {16,8,375},
                                                ge::FORMAT_ND, {{15, 16},{8,8},{375,375}});
    auto tensor_desc_dout = create_desc_shape_range({1},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {1},
                                                ge::FORMAT_ND, {{1, 1}});
    op.UpdateInputDesc("predict", tensor_desc);
    op.UpdateInputDesc("label", tensor_desc);
    op.UpdateInputDesc("dout", tensor_desc_dout);
    op.SetAttr("reduction", "mean");
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {-1, 8, 375};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{15, 16},{8,8},{375,375}};
    EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(mse_loss_grad, mse_loss_grad_infershape_diff_test2){
    ge::op::MseLossGrad op;
    auto tensor_desc_predict = create_desc_shape_range({-1,8,375},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {16,8,375},
                                                ge::FORMAT_ND, {{15, 16},{8,8},{375,375}});
    auto tensor_desc_label = create_desc_shape_range({-1,8,375},
                                                ge::DT_FLOAT, ge::FORMAT_ND,
                                                {16,8,375},
                                                ge::FORMAT_ND, {{15, 16},{8,8},{375,375}});
    auto tensor_desc_dout = create_desc_shape_range({1},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {1},
                                                ge::FORMAT_ND, {{1, 1}});
    op.UpdateInputDesc("predict", tensor_desc_predict);
    op.UpdateInputDesc("label", tensor_desc_label);
    op.UpdateInputDesc("dout", tensor_desc_dout);
    op.SetAttr("reduction", "mean");
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
