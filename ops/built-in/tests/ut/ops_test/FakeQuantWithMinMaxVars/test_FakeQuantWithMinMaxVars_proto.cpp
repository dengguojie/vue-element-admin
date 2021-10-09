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
 * @file test_FakeQuantWithMinMaxVars_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>  // NOLINT
#include <iostream>  // NOLINT
#include "op_proto_test_util.h"  // NOLINT
#include "elewise_calculation_ops.h"  // NOLINT

class fake_quant_with_min_max_vars : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "fake_quant_with_min_max_vars SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "fake_quant_with_min_max_vars TearDown" << std::endl;
  }
};

TEST_F(fake_quant_with_min_max_vars, fake_quant_with_min_max_vars_success_case) {
    ge::op::FakeQuantWithMinMaxVars op;

    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 100}, {1, 100}};
    auto input_tensor_desc = create_desc_shape_range({-1, -1},
                                               ge::DT_FLOAT, ge::FORMAT_ND,
                                               {-1, -1},
                                               ge::FORMAT_ND, shape_range);

    auto minmax_tensor_desc = create_desc_shape_range({1},
                                                 ge::DT_FLOAT, ge::FORMAT_ND,
                                                 {1},
                                                 ge::FORMAT_ND, {{1, 1}});

    op.UpdateInputDesc("x", input_tensor_desc);
    op.UpdateInputDesc("min", minmax_tensor_desc);
    op.UpdateInputDesc("max", minmax_tensor_desc);
    int8_t num_bits = 8;
    op.SetAttr("num_bits", num_bits);
    bool narrow_range = false;
    op.SetAttr("narrow_range", narrow_range);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto ret2 = op.VerifyAllAttr(true);
    EXPECT_EQ(ret2, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {-1, -1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_shape_range, shape_range);
}

TEST_F(fake_quant_with_min_max_vars, fake_quant_with_min_max_vars_failed_case) {
    ge::op::FakeQuantWithMinMaxVars op;

    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 100}, {1, 100}};
    auto input_tensor_desc = create_desc_shape_range({-1, -1},
                                               ge::DT_FLOAT16, ge::FORMAT_ND,
                                               {-1, -1},
                                               ge::FORMAT_ND, shape_range);

    auto min_tensor_desc = create_desc_shape_range({1, 1},
                                                 ge::DT_FLOAT16, ge::FORMAT_ND,
                                                 {1},
                                                 ge::FORMAT_ND, {{1, 1}, {1, 1}});

    auto max_tensor_desc = create_desc_shape_range({1},
                                                 ge::DT_FLOAT16, ge::FORMAT_ND,
                                                 {1},
                                                 ge::FORMAT_ND, {{1, 1}});

    op.UpdateInputDesc("x", input_tensor_desc);
    op.UpdateInputDesc("min", min_tensor_desc);
    op.UpdateInputDesc("max", max_tensor_desc);
    int8_t num_bits = 8;
    op.SetAttr("num_bits", num_bits);
    bool narrow_range = false;
    op.SetAttr("narrow_range", narrow_range);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto ret2 = op.VerifyAllAttr(true);
    EXPECT_EQ(ret2, ge::GRAPH_FAILED);

}

TEST_F(fake_quant_with_min_max_vars, fake_quant_with_min_max_vars_failed_case_2) {
    ge::op::FakeQuantWithMinMaxVars op;

    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 100}, {1, 100}};
    auto input_tensor_desc = create_desc_shape_range({-1, -1},
                                               ge::DT_FLOAT16, ge::FORMAT_ND,
                                               {-1, -1},
                                               ge::FORMAT_ND, shape_range);

    auto min_tensor_desc = create_desc_shape_range({1},
                                                 ge::DT_FLOAT16, ge::FORMAT_ND,
                                                 {1},
                                                 ge::FORMAT_ND, {{1, 1}});

    auto max_tensor_desc = create_desc_shape_range({2},
                                                 ge::DT_FLOAT16, ge::FORMAT_ND,
                                                 {1},
                                                 ge::FORMAT_ND, {{2, 2}});

    op.UpdateInputDesc("x", input_tensor_desc);
    op.UpdateInputDesc("min", min_tensor_desc);
    op.UpdateInputDesc("max", max_tensor_desc);
    int8_t num_bits = 8;
    op.SetAttr("num_bits", num_bits);
    bool narrow_range = false;
    op.SetAttr("narrow_range", narrow_range);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto ret2 = op.VerifyAllAttr(true);
    EXPECT_EQ(ret2, ge::GRAPH_FAILED);

}