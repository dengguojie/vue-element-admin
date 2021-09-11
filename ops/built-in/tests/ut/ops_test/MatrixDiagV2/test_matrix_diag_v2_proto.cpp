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
 * @file test_matrix_diag_v2_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"
#include "split_combination_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "array_ops.h"

class MatrixDiagV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MatrixDiagV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MatrixDiagV2 TearDown" << std::endl;
  }
};

TEST_F(MatrixDiagV2, matrix_diag_v2_infer_shape) {
  ge::op::MatrixDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("diagonal", tensor_desc1);
  op.UpdateInputDesc("k", tensor_desc1);
  op.UpdateInputDesc("num_rows", tensor_desc2);
  op.UpdateInputDesc("num_cols", tensor_desc2);
  op.UpdateInputDesc("padding_value", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MatrixDiagV2, matrix_diag_v2_infer_diagonal_shape_failed) {
  ge::op::MatrixDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("diagonal", tensor_desc2);
  op.UpdateInputDesc("k", tensor_desc1);
  op.UpdateInputDesc("num_rows", tensor_desc2);
  op.UpdateInputDesc("num_cols", tensor_desc2);
  op.UpdateInputDesc("padding_value", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixDiagV2, matrix_diag_v2_infer_k_shape_failed) {
  ge::op::MatrixDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("diagonal", tensor_desc1);
  op.UpdateInputDesc("k", tensor_desc2);
  op.UpdateInputDesc("num_rows", tensor_desc2);
  op.UpdateInputDesc("num_cols", tensor_desc2);
  op.UpdateInputDesc("padding_value", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixDiagV2, matrix_diag_v2_infer_num_rows_shape_failed) {
  ge::op::MatrixDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("diagonal", tensor_desc1);
  op.UpdateInputDesc("k", tensor_desc1);
  op.UpdateInputDesc("num_rows", tensor_desc1);
  op.UpdateInputDesc("num_cols", tensor_desc2);
  op.UpdateInputDesc("padding_value", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixDiagV2, matrix_diag_v2_infer_num_cols_shape_failed) {
  ge::op::MatrixDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("diagonal", tensor_desc1);
  op.UpdateInputDesc("k", tensor_desc1);
  op.UpdateInputDesc("num_rows", tensor_desc2);
  op.UpdateInputDesc("num_cols", tensor_desc1);
  op.UpdateInputDesc("padding_value", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixDiagV2, matrix_diag_v2_infer_padding_value_shape_failed) {
  ge::op::MatrixDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("diagonal", tensor_desc1);
  op.UpdateInputDesc("k", tensor_desc1);
  op.UpdateInputDesc("num_rows", tensor_desc2);
  op.UpdateInputDesc("num_cols", tensor_desc2);
  op.UpdateInputDesc("padding_value", tensor_desc1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixDiagV2, matrix_diag_v2_infer_shape_k_num_failed) {
  ge::op::MatrixDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("diagonal", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT32);
  const_desc.SetOriginShape(ge::Shape({3}));
  int32_t const_value[3] = {7, 6, 5};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 3 * sizeof(int32_t)));
  op.set_input_k(const_op);
  op.UpdateInputDesc("k", const_desc);

  op.UpdateInputDesc("num_rows", tensor_desc2);
  op.UpdateInputDesc("num_cols", tensor_desc2);
  op.UpdateInputDesc("padding_value", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixDiagV2, matrix_diag_v2_infer_shape_k_val_failed) {
  ge::op::MatrixDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("diagonal", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  const_desc.SetOriginShape(ge::Shape({2}));
  int32_t const_value[2] = {7, 6};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 2 * sizeof(int32_t)));
  op.set_input_k(const_op);
  op.UpdateInputDesc("k", const_desc);

  op.UpdateInputDesc("num_rows", tensor_desc2);
  op.UpdateInputDesc("num_cols", tensor_desc2);
  op.UpdateInputDesc("padding_value", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixDiagV2, matrix_diag_v2_infer_shape_k_val_failed2) {
  ge::op::MatrixDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("diagonal", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  const_desc.SetOriginShape(ge::Shape({2}));
  int32_t const_value[2] = {6, 7};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 2 * sizeof(int32_t)));
  op.set_input_k(const_op);
  op.UpdateInputDesc("k", const_desc);

  op.UpdateInputDesc("num_rows", tensor_desc2);
  op.UpdateInputDesc("num_cols", tensor_desc2);
  op.UpdateInputDesc("padding_value", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixDiagV2, matrix_diag_v2_infer_shape_k_val_failed3) {
  ge::op::MatrixDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  ge::TensorDesc const_desc0(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value0[2] = {2, 7};
  auto const_op0 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc0, (uint8_t *)const_value0, 2 * sizeof(int32_t)));
  op.set_input_diagonal(const_op0);
  op.UpdateInputDesc("diagonal", const_desc0);

  ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[2] = {0, 0};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 2 * sizeof(int32_t)));
  op.set_input_k(const_op);
  op.UpdateInputDesc("k", const_desc);

  ge::TensorDesc const_desc1(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value1[1] = {1};
  auto const_op1 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc1, (uint8_t *)const_value1, 1 * sizeof(int32_t)));
  op.set_input_num_rows(const_op1);
  op.UpdateInputDesc("num_rows", const_desc1);
  op.UpdateInputDesc("num_cols", tensor_desc2);
  op.UpdateInputDesc("padding_value", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixDiagV2, matrix_diag_v2_infer_shape_k_val_failed4) {
  ge::op::MatrixDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  ge::TensorDesc const_desc0(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value0[2] = {2, 7};
  auto const_op0 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc0, (uint8_t *)const_value0, 2 * sizeof(int32_t)));
  op.set_input_diagonal(const_op0);
  op.UpdateInputDesc("diagonal", const_desc0);

  ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[2] = {0, 0};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 2 * sizeof(int32_t)));
  op.set_input_k(const_op);
  op.UpdateInputDesc("k", const_desc);

  ge::TensorDesc const_desc1(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value1[1] = {3};
  auto const_op1 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc1, (uint8_t *)const_value1, 1 * sizeof(int32_t)));
  op.set_input_num_rows(const_op1);
  op.UpdateInputDesc("num_rows", const_desc1);

  ge::TensorDesc const_desc2(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value2[1] = {0};
  auto const_op2 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc2, (uint8_t *)const_value2, 1 * sizeof(int32_t)));
  op.set_input_num_cols(const_op2);
  op.UpdateInputDesc("num_cols", const_desc2);
  op.UpdateInputDesc("padding_value", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixDiagV2, matrix_diag_v2_infer_shape_k_val_failed5) {
  ge::op::MatrixDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  ge::TensorDesc const_desc0(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value0[2] = {2, 7};
  auto const_op0 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc0, (uint8_t *)const_value0, 2 * sizeof(int32_t)));
  op.set_input_diagonal(const_op0);
  op.UpdateInputDesc("diagonal", const_desc0);

  ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[2] = {0, 0};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 2 * sizeof(int32_t)));
  op.set_input_k(const_op);
  op.UpdateInputDesc("k", const_desc);

  ge::TensorDesc const_desc1(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value1[1] = {3};
  auto const_op1 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc1, (uint8_t *)const_value1, 1 * sizeof(int32_t)));
  op.set_input_num_rows(const_op1);
  op.UpdateInputDesc("num_rows", const_desc1);

  ge::TensorDesc const_desc2(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value2[1] = {3};
  auto const_op2 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc2, (uint8_t *)const_value2, 1 * sizeof(int32_t)));
  op.set_input_num_cols(const_op2);
  op.UpdateInputDesc("num_cols", const_desc2);
  op.UpdateInputDesc("padding_value", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixDiagV2, matrix_diag_v2_infer_shape_k_val_success) {
  ge::op::MatrixDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  ge::TensorDesc const_desc0(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  const_desc0.SetOriginShape(ge::Shape({2}));
  int32_t const_value0[2] = {2, 7};
  auto const_op0 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc0, (uint8_t *)const_value0, 2 * sizeof(int32_t)));
  op.set_input_diagonal(const_op0);
  op.UpdateInputDesc("diagonal", const_desc0);

  ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  const_desc.SetOriginShape(ge::Shape({2}));
  int32_t const_value[2] = {0, 0};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 2 * sizeof(int32_t)));
  op.set_input_k(const_op);
  op.UpdateInputDesc("k", const_desc);

  ge::TensorDesc const_desc1(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  const_desc1.SetOriginShape(ge::Shape());
  int32_t const_value1[1] = {2};
  auto const_op1 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc1, (uint8_t *)const_value1, 1 * sizeof(int32_t)));
  op.set_input_num_rows(const_op1);
  op.UpdateInputDesc("num_rows", const_desc1);

  ge::TensorDesc const_desc2(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  const_desc2.SetOriginShape(ge::Shape());
  int32_t const_value2[1] = {2};
  auto const_op2 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc2, (uint8_t *)const_value2, 1 * sizeof(int32_t)));
  op.set_input_num_cols(const_op2);
  op.UpdateInputDesc("num_cols", const_desc2);
  op.UpdateInputDesc("padding_value", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}