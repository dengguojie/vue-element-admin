/**
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_concat_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"

class UnravelIndex : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UnravelIndex SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "UnravelIndex TearDown" << std::endl;
  }
};

TEST_F(UnravelIndex, unravel_index_infer_shape01) {
  ge::op::UnravelIndex op;
  std::vector<std::pair<int64_t,int64_t>> dim_shape_range = {{2, 2}};
  auto dim_tensor_desc = create_desc_shape_range({2},
                                                 ge::DT_INT32, ge::FORMAT_ND,
                                                 {2},
                                                 ge::FORMAT_ND, dim_shape_range);
  std::vector<std::pair<int64_t,int64_t>> indice_shape_range = {{2,2}};
  auto indice_tensor_desc = create_desc_shape_range({2},
                                                    ge::DT_INT32, ge::FORMAT_ND,
                                                    {2},
                                                    ge::FORMAT_ND, indice_shape_range);
  op.UpdateInputDesc("indices", indice_tensor_desc);
  op.UpdateInputDesc("dims", dim_tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(UnravelIndex, unravel_index_infer_shape02) {
  ge::op::UnravelIndex op;
  std::vector<std::pair<int64_t,int64_t>> dim_shape_range = {{2, 2}};
  auto dim_tensor_desc = create_desc_shape_range({2},
                                                 ge::DT_INT32, ge::FORMAT_ND,
                                                 {2},
                                                 ge::FORMAT_ND, dim_shape_range);
  std::vector<std::pair<int64_t,int64_t>> indice_shape_range = {{-1, 1}};
  auto indice_tensor_desc = create_desc_shape_range({-1},
                                                    ge::DT_INT32, ge::FORMAT_ND,
                                                    {2},
                                                    ge::FORMAT_ND, indice_shape_range);
  op.UpdateInputDesc("indices", indice_tensor_desc);
  op.UpdateInputDesc("dims", dim_tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(UnravelIndex, unravel_index_infer_shape03) {
  ge::op::UnravelIndex op;
  std::vector<std::pair<int64_t,int64_t>> dim_shape_range = {{-1, 1}};
  auto dim_tensor_desc = create_desc_shape_range({-1},
                                                 ge::DT_INT32, ge::FORMAT_ND,
                                                 {2},
                                                 ge::FORMAT_ND, dim_shape_range);
  std::vector<std::pair<int64_t,int64_t>> indice_shape_range = {{1, 1}};
  auto indice_tensor_desc = create_desc_shape_range({1},
                                                    ge::DT_INT32, ge::FORMAT_ND,
                                                    {1},
                                                    ge::FORMAT_ND, indice_shape_range);
  op.UpdateInputDesc("indices", indice_tensor_desc);
  op.UpdateInputDesc("dims", dim_tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(UnravelIndex, unravel_index_infer_shape04) {
  ge::op::UnravelIndex op;
  std::vector<std::pair<int64_t,int64_t>> dim_shape_range = {{2, 2}, {1, 1}};
  auto dim_tensor_desc = create_desc_shape_range({2, 1},
                                                 ge::DT_INT32, ge::FORMAT_ND,
                                                 {2, 1},
                                                 ge::FORMAT_ND, dim_shape_range);
  std::vector<std::pair<int64_t,int64_t>> indice_shape_range = {{-1, 1}};
  auto indice_tensor_desc = create_desc_shape_range({-1},
                                                    ge::DT_INT32, ge::FORMAT_ND,
                                                    {2},
                                                    ge::FORMAT_ND, indice_shape_range);
  op.UpdateInputDesc("indices", indice_tensor_desc);
  op.UpdateInputDesc("dims", dim_tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}