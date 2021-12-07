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
 * @file test_fast_gelu_v2_proto.cpp
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

class fast_gelu_v2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "fast_gelu_v2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "fast_gelu_v2 TearDown" << std::endl;
  }
};

TEST_F(fast_gelu_v2, fast_gelu_infershape_test) {
  ge::op::FastGeluV2 op;

  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 16}, {1, 16}};

  auto tensor_desc = create_desc_shape_range({-1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {16, 16},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1, 16}, {1, 16}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
