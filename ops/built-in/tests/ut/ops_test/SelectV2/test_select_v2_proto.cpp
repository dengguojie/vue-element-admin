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
 * @file test_select_v2_proto.cpp
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class select_v2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "select_v2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "select_v2 TearDown" << std::endl;
  }
};

TEST_F(select_v2, select_v2_infer_shape_2) {
  ge::op::SelectV2 op;

  std::vector<std::pair<int64_t, int64_t>> shape_range_con = {{0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}};
  auto tensor_desc_con = create_desc_shape_range({1, 1, 1, 1, 1},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {1, 1, 1, 1, 1},
                                             ge::FORMAT_ND, shape_range_con);
  op.UpdateInputDesc("condition", tensor_desc_con);

  std::vector<std::pair<int64_t, int64_t>> shape_range_x1 = {{0, 1}, {1, 2}, {2, 3}};
  auto tensor_desc_x1 = create_desc_shape_range({1, 1, 1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {1, 1, 1},
                                             ge::FORMAT_ND, shape_range_x1);
  op.UpdateInputDesc("then", tensor_desc_x1);

  std::vector<std::pair<int64_t, int64_t>> shape_range_x2 = {{1, 3}, {2, 5}};
  auto tensor_desc_x2 = create_desc_shape_range({1, 1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {1, 1},
                                             ge::FORMAT_ND, shape_range_x2);
  op.UpdateInputDesc("else", tensor_desc_x2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("result");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {1, 1, 1, 1, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {
      {0, 1}, {0, 1}, {0, 1}, {0, 3}, {0, 5}
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
