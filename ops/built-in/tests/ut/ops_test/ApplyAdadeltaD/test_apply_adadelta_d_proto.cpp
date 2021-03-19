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
 * @file test_dynamic_apply_adadelta_d_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>  // NOLINT
#include <iostream>
#include "op_proto_test_util.h"  // NOLINT
#include "nn_training_ops.h"  // NOLINT

class apply_adadelta_d : public testing::Test {
 protected:
    static void SetUpTestCase() {
        std::cout << "apply_adadelta_d SetUp" << std::endl;
    }

    static void TearDownTestCase() {
    std::cout << "apply_adadelta_d Test TearDown" << std::endl;
  }
};

TEST_F(apply_adadelta_d, cce_apply_adadelta_d_infer_shape_success) {
    ge::op::ApplyAdadeltaD op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}};
    auto tensor_desc = create_desc_shape_range({-1},
                                               ge::DT_FLOAT16, ge::FORMAT_ND,
                                               {64},
                                               ge::FORMAT_ND, shape_range);
    auto tensor_desc1 = create_desc_shape_range({-1},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {1},
                                                ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("var", tensor_desc);
    op.UpdateInputDesc("accum", tensor_desc);
    op.UpdateInputDesc("accum_update", tensor_desc);
    op.UpdateInputDesc("grad", tensor_desc);
    op.UpdateInputDesc("lr", tensor_desc);
    op.UpdateInputDesc("rho", tensor_desc);
    op.UpdateInputDesc("epsilon", tensor_desc);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("var");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {-1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_desc.GetOriginShape().GetDims(), expected_output_shape);
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{2, 100}};
    EXPECT_EQ(output_shape_range, expected_shape_range);

    auto accum_out_desc = op.GetOutputDesc("accum");
    EXPECT_EQ(accum_out_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_accum_out_shape = {-1};
    EXPECT_EQ(accum_out_desc.GetShape().GetDims(), expected_accum_out_shape);
    EXPECT_EQ(accum_out_desc.GetOriginShape().GetDims(), expected_accum_out_shape);
    std::vector<std::pair<int64_t, int64_t>> accum_out_shape_range;
    EXPECT_EQ(accum_out_desc.GetShapeRange(accum_out_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_accum_out_shape_range = {{2, 100}};
    EXPECT_EQ(accum_out_shape_range, expected_accum_out_shape_range);

    auto accum_update_out_desc = op.GetOutputDesc("accum_update");
    EXPECT_EQ(accum_update_out_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_accum_update_out_shape = {-1};
    EXPECT_EQ(accum_update_out_desc.GetShape().GetDims(), expected_accum_update_out_shape);
    EXPECT_EQ(accum_update_out_desc.GetOriginShape().GetDims(), expected_accum_update_out_shape);
    std::vector<std::pair<int64_t, int64_t>> accum_update_out_shape_range;
    EXPECT_EQ(accum_update_out_desc.GetShapeRange(accum_update_out_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_accum_update_out_shape_range = {{2, 100}};
    EXPECT_EQ(accum_update_out_shape_range, expected_accum_update_out_shape_range);
}
