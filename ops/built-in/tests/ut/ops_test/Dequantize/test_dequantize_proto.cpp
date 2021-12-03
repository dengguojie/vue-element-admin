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
 * @file test_quantize_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>  // NOLINT
#include <iostream>  // NOLINT
#include "op_proto_test_util.h"  // NOLINT
#include "quantize_ops.h"  // NOLINT

class dequantize : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dequantize SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dequantize TearDown" << std::endl;
  }
};

TEST_F(dequantize, dequantize_case) {
    ge::op::Dequantize op;

    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}, {2, 100}};
    auto tensor_desc = create_desc_shape_range({-1, -1},
                                               ge::DT_INT8, ge::FORMAT_ND,
                                               {11, 33},
                                               ge::FORMAT_ND, shape_range);

    auto tensor_desc_1 = create_desc_shape_range({-1},
                                                 ge::DT_FLOAT, ge::FORMAT_ND,
                                                 {1, },
                                                 ge::FORMAT_ND, {{1, 10}});
    op.UpdateInputDesc("x", tensor_desc);
    op.UpdateInputDesc("min_range", tensor_desc_1);
    op.UpdateInputDesc("max_range", tensor_desc_1);
    std::string mode = "SCALED";
    op.SetAttr("mode", mode);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {-1, -1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{2, 100}, {2, 100}};
    EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(dequantize, dequantize_case_001) {
  ge::op::Dequantize op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}, {2, 100}};
  auto tensor_desc =
      create_desc_shape_range({-1, -1}, ge::DT_INT8, ge::FORMAT_ND, {11, 33}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  std::string mode = "MIN_SECOND";
  op.SetAttr("mode", mode);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}