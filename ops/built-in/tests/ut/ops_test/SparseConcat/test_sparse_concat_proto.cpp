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
 * @file test_sparse_concat_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "sparse_ops.h"
class SparseConcatTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseConcat SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseConcat TearDown" << std::endl;
  }
};

TEST_F(SparseConcatTest, InferShape_01) {
  ge::op::SparseConcat op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  op.SetAttr("N", 2);
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

 const int32_t size = 2;

  op.create_dynamic_input_indices(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("indices", i, create_desc({3}, ge::DT_INT64));
  }

  op.create_dynamic_input_values(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("values", i, create_desc({2, 3}, ge::DT_INT64));
  }

  op.create_dynamic_input_shapes(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("shapes", i, create_desc({2, 3}, ge::DT_INT64));
  }
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("indices", i, create_desc({4, 3}, ge::DT_INT64));
  }
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("values", i, create_desc({-2}, ge::DT_INT64));
  }
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("shapes", i, create_desc({2}, ge::DT_INT64));
  }
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("indices", i, create_desc({-2, 3}, ge::DT_INT64));
  }
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("indices", i, create_desc({2, 3}, ge::DT_INT64));
  }
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("values", i, create_desc({2}, ge::DT_INT64));
  }
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
 }