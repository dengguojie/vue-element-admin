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
 * @file test_decode_jpeg_proto.cpp
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

class SparseCross : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseCross SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseCross TearDown" << std::endl;
  }
};

TEST_F(SparseCross, sparse_cross_infer_shape_01) {
  ge::op::SparseCross op;

  op.SetAttr("out_type", ge::DT_INT64);
  op.SetAttr("internal_type", ge::DT_INT64);
  const int32_t size = 2;
  op.create_dynamic_input_indices(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("indices", i, create_desc({2}, ge::DT_INT64));
  }

  op.create_dynamic_input_values(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("values", i, create_desc({2}, ge::DT_INT64));
  }

  op.create_dynamic_input_shapes(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("shapes", i, create_desc({2}, ge::DT_INT64));
  }

  op.create_dynamic_input_dense_inputs(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("dense_inputs", i, create_desc({2}, ge::DT_INT64));
  }

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseCross, sparse_cross_infer_shape_02) {
  ge::op::SparseCross op;

  op.SetAttr("out_type", ge::DT_INT64);
  op.SetAttr("internal_type", ge::DT_INT64);
  const int32_t size = 2;
  op.create_dynamic_input_indices(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("indices", i, create_desc({2, 2}, ge::DT_INT64));
  }

  op.create_dynamic_input_values(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("values", i, create_desc({2, 2}, ge::DT_INT64));
  }

  op.create_dynamic_input_shapes(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("shapes", i, create_desc({2}, ge::DT_INT64));
  }

  op.create_dynamic_input_dense_inputs(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("dense_inputs", i, create_desc({2}, ge::DT_INT64));
  }

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseCross, sparse_cross_infer_shape_03) {
  ge::op::SparseCross op;

  op.SetAttr("out_type", ge::DT_INT64);
  op.SetAttr("internal_type", ge::DT_INT64);
  const int32_t size = 2;
  op.create_dynamic_input_indices(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("indices", i, create_desc({2, 2}, ge::DT_INT64));
  }

  op.create_dynamic_input_values(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("values", i, create_desc({2}, ge::DT_INT64));
  }

  op.create_dynamic_input_shapes(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("shapes", i, create_desc({2, 2}, ge::DT_INT64));
  }

  op.create_dynamic_input_dense_inputs(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("dense_inputs", i, create_desc({2}, ge::DT_INT64));
  }

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseCross, sparse_cross_infer_04) {
  ge::op::SparseCross op;

  op.SetAttr("out_type", ge::DT_FLOAT);
  op.SetAttr("internal_type", ge::DT_INT64);
  const int32_t size = 2;
  op.create_dynamic_input_indices(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("indices", i, create_desc({2, 2}, ge::DT_INT64));
  }

  op.create_dynamic_input_values(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("values", i, create_desc({2}, ge::DT_INT64));
  }

  op.create_dynamic_input_shapes(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("shapes", i, create_desc({2}, ge::DT_INT64));
  }

  op.create_dynamic_input_dense_inputs(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("dense_inputs", i, create_desc({2}, ge::DT_INT64));
  }

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseCross, sparse_cross_infer_05) {
  ge::op::SparseCross op;

  op.SetAttr("out_type", ge::DT_INT64);
  op.SetAttr("internal_type", ge::DT_FLOAT);
  const int32_t size = 2;
  op.create_dynamic_input_indices(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("indices", i, create_desc({2, 2}, ge::DT_INT64));
  }

  op.create_dynamic_input_values(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("values", i, create_desc({2}, ge::DT_INT64));
  }

  op.create_dynamic_input_shapes(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("shapes", i, create_desc({2}, ge::DT_INT64));
  }

  op.create_dynamic_input_dense_inputs(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("dense_inputs", i, create_desc({2}, ge::DT_INT64));
  }

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}