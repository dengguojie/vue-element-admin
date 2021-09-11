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
 * @file test_matrix_set_diag_v2_proto.cpp
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

class MatrixSetDiagV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MatrixSetDiagV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MatrixSetDiagV2 TearDown" << std::endl;
  }
};

TEST_F(MatrixSetDiagV2, matrix_set_diag_v2_infer_shape) {
  ge::op::MatrixSetDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range1 = {{2, 2}, {2, 2}};
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2, 2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2, 2},
                                              ge::FORMAT_ND, shape_range1);
  auto tensor_desc2 = create_desc_shape_range({2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2},
                                              ge::FORMAT_ND, shape_range2);
  auto tensor_desc3 = create_desc_shape_range({},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {},
                                              ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("input", tensor_desc1);
  op.UpdateInputDesc("diagonal", tensor_desc2);
  op.UpdateInputDesc("k", tensor_desc3);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MatrixSetDiagV2, matrix_set_diag_v2_infer_shape_check_input0_failed) {
  ge::op::MatrixSetDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range1 = {{2, 2}, {2, 2}};
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2},
                                              ge::FORMAT_ND, shape_range1);
  op.UpdateInputDesc("input", tensor_desc1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2, matrix_set_diag_v2_infer_shape_check_input1_failed) {
  ge::op::MatrixSetDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range1 = {{2, 2}, {2, 2}};
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2, 2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2, 2},
                                              ge::FORMAT_ND, shape_range1);

  op.UpdateInputDesc("input", tensor_desc1);
  auto tensor_desc2 = create_desc_shape_range({2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2},
                                              ge::FORMAT_ND, shape_range2);
  op.UpdateInputDesc("diagonal", tensor_desc2);

  auto tensor_desc3 = create_desc_shape_range({2,2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2,2},
                                              ge::FORMAT_ND, shape_range1);
  op.UpdateInputDesc("k", tensor_desc3);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2, matrix_set_diag_v2_infer_shape_check_input2_failed) {
  ge::op::MatrixSetDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range1 = {{2, 2}, {2, 2}};
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2, 2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2, 2},
                                              ge::FORMAT_ND, shape_range1);
  op.UpdateInputDesc("input", tensor_desc1);

  auto tensor_desc2 = create_desc_shape_range({2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2},
                                              ge::FORMAT_ND, shape_range1);
  op.UpdateInputDesc("padding_value", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2, matrix_set_diag_v2_infer_shape_check_k_num_failed) {
  ge::op::MatrixSetDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range1 = {{2, 2}, {2, 2}};
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2, 2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2, 2},
                                              ge::FORMAT_ND, shape_range1);

  op.UpdateInputDesc("input", tensor_desc1);
  auto tensor_desc2 = create_desc_shape_range({2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2},
                                              ge::FORMAT_ND, shape_range2);
  op.UpdateInputDesc("diagonal", tensor_desc2);

  ge::TensorDesc const_desc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[3] = {3, 2, 1};
  auto const_op = ge::op::Constant().set_attr_value(
    ge::Tensor(const_desc, (uint8_t *)const_value, 3 * sizeof(int32_t)));
  op.set_input_k(const_op);
  op.UpdateInputDesc("k", const_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2, matrix_set_diag_v2_infer_shape_check_k_value_failed) {
  ge::op::MatrixSetDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range1 = {{2, 2}, {2, 2}};
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2, 2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2, 2},
                                              ge::FORMAT_ND, shape_range1);

  op.UpdateInputDesc("input", tensor_desc1);
  auto tensor_desc2 = create_desc_shape_range({2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2},
                                              ge::FORMAT_ND, shape_range2);
  op.UpdateInputDesc("diagonal", tensor_desc2);

  ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[2] = {3, 2};
  auto const_op = ge::op::Constant().set_attr_value(
    ge::Tensor(const_desc, (uint8_t *)const_value, 2 * sizeof(int32_t)));
  op.set_input_k(const_op);
  op.UpdateInputDesc("k", const_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2, matrix_set_diag_v2_infer_shape_check_diagonal_rank_failed) {
  ge::op::MatrixSetDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range1 = {{2, 2}, {2, 2}};
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2, 2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2, 2},
                                              ge::FORMAT_ND, shape_range1);

  op.UpdateInputDesc("input", tensor_desc1);
  auto tensor_desc2 = create_desc_shape_range({2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2},
                                              ge::FORMAT_ND, shape_range2);
  op.UpdateInputDesc("diagonal", tensor_desc2);

  ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  const_desc.SetOriginShape(ge::Shape({2}));
  int32_t const_value[2] = {1, 2};
  auto const_op = ge::op::Constant().set_attr_value(
    ge::Tensor(const_desc, (uint8_t *)const_value, 2 * sizeof(int32_t)));
  op.set_input_k(const_op);
  op.UpdateInputDesc("k", const_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2, matrix_set_diag_v2_infer_shape_check_diagonal_rank_failed2) {
  ge::op::MatrixSetDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range1 = {{2, 2}, {2, 2}};
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2, 2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2, 2},
                                              ge::FORMAT_ND, shape_range1);

  op.UpdateInputDesc("input", tensor_desc1);
  auto tensor_desc2 = create_desc_shape_range({},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {},
                                              ge::FORMAT_ND, {{}});
  op.UpdateInputDesc("diagonal", tensor_desc2);

  auto tensor_desc3 = create_desc_shape_range({},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {},
                                              ge::FORMAT_ND, {{}});
  op.UpdateInputDesc("k", tensor_desc3);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2, matrix_set_diag_v2_infer_shape_check_k_value_failed2) {
  ge::op::MatrixSetDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range1 = {{2, 2}, {2, 2}};
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2, 2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2, 2},
                                              ge::FORMAT_ND, shape_range1);

  op.UpdateInputDesc("input", tensor_desc1);
  auto tensor_desc2 = create_desc_shape_range({2,2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2,2},
                                              ge::FORMAT_ND, shape_range1);
  op.UpdateInputDesc("diagonal", tensor_desc2);

  ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[2] = {5, 6};
  auto const_op = ge::op::Constant().set_attr_value(
    ge::Tensor(const_desc, (uint8_t *)const_value, 2 * sizeof(int32_t)));
  op.set_input_k(const_op);
  op.UpdateInputDesc("k", const_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2, matrix_set_diag_v2_infer_shape_check_k_value_failed3) {
  ge::op::MatrixSetDiagV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range1 = {{2, 2}, {2, 2}};
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2, 2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2, 2},
                                              ge::FORMAT_ND, shape_range1);

  op.UpdateInputDesc("input", tensor_desc1);
  auto tensor_desc2 = create_desc_shape_range({2,2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2,2},
                                              ge::FORMAT_ND, shape_range1);
  op.UpdateInputDesc("diagonal", tensor_desc2);

  ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[2] = {1, 6};
  auto const_op = ge::op::Constant().set_attr_value(
    ge::Tensor(const_desc, (uint8_t *)const_value, 2 * sizeof(int32_t)));
  op.set_input_k(const_op);
  op.UpdateInputDesc("k", const_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}