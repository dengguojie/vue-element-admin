/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_scatter_nd_d_proto.cpp
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

class scatter_nd_d : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "scatter_nd_d SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "scatter_nd_d TearDown" << std::endl;
  }
};

TEST_F(scatter_nd_d, scatter_nd_infershape_diff_test_1) {
  ge::op::ScatterNdD op;
  op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2}, ge::FORMAT_ND));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(scatter_nd_d, scatter_nd_infershape_diff_test_2) {
  ge::op::ScatterNdD op;
  op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2}, ge::FORMAT_ND));

  std::vector<int64_t> shape = {2, 3};
  op.SetAttr("shape", shape);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(scatter_nd_d, scatter_nd_infershape_diff_test_3) {
  ge::op::ScatterNdD op;
  op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2}, ge::FORMAT_ND));

  std::vector<int64_t> shape = {2, 3, 4};
  op.SetAttr("shape", shape);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(scatter_nd_d, scatter_nd_infershape_diff_test_4) {
  ge::op::ScatterNdD op;
  op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2}, ge::FORMAT_ND));

  std::vector<int64_t> shape = {2, 2, 2};
  op.SetAttr("shape", shape);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
