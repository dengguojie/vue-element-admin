/**
 * Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_isotonic_regression_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>

#include "array_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"

#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"

class IsotonicRegressionTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "IsotonicRegression test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "IsotonicRegression test TearDown" << std::endl;
  }
};

TEST_F(IsotonicRegressionTest, IsotonicRegression_test_case_float16){
  ge::op::IsotonicRegression op;
  op.UpdateInputDesc("input", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.SetAttr("output_dtype", ge::DT_FLOAT16);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("output");
  auto segments_desc = op.GetOutputDesc("segments");
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(segments_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(segments_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(IsotonicRegressionTest, IsotonicRegression_test_case_float32){
  ge::op::IsotonicRegression op;
  op.UpdateInputDesc("input", create_desc({1, 2}, ge::DT_FLOAT));
  op.SetAttr("output_dtype", ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("output");
  auto segments_desc = op.GetOutputDesc("segments");
  std::vector<int64_t> expected_output_shape = {1, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(segments_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(segments_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(IsotonicRegressionTest, IsotonicRegression_test_case_float64){
  ge::op::IsotonicRegression op;
  op.UpdateInputDesc("input", create_desc({1, 2, 3, 4}, ge::DT_DOUBLE));
  op.SetAttr("output_dtype", ge::DT_DOUBLE);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("output");
  auto segments_desc = op.GetOutputDesc("segments");
  std::vector<int64_t> expected_output_shape = {1, 2, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(segments_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_DOUBLE);
  EXPECT_EQ(segments_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(IsotonicRegressionTest, IsotonicRegression_test_case_uint8){
  ge::op::IsotonicRegression op;
  op.UpdateInputDesc("input", create_desc({2}, ge::DT_UINT8));
  op.SetAttr("output_dtype", ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("output");
  auto segments_desc = op.GetOutputDesc("segments");
  std::vector<int64_t> expected_output_shape = {2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(segments_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(segments_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(IsotonicRegressionTest, IsotonicRegression_test_case_int8){
  ge::op::IsotonicRegression op;
  op.UpdateInputDesc("input", create_desc({}, ge::DT_INT8));
  op.SetAttr("output_dtype", ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("output");
  auto segments_desc = op.GetOutputDesc("segments");
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(segments_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(segments_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(IsotonicRegressionTest, IsotonicRegression_test_case_uint16){
  ge::op::IsotonicRegression op;
  op.UpdateInputDesc("input", create_desc({4, 3, 4}, ge::DT_UINT16));
  op.SetAttr("output_dtype", ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("output");
  auto segments_desc = op.GetOutputDesc("segments");
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(segments_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(segments_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(IsotonicRegressionTest, IsotonicRegression_test_case_int16){
  ge::op::IsotonicRegression op;
  op.UpdateInputDesc("input", create_desc({1, 2, 3}, ge::DT_INT16));
  op.SetAttr("output_dtype", ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("output");
  auto segments_desc = op.GetOutputDesc("segments");
  std::vector<int64_t> expected_output_shape = {1, 2, 3};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(segments_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(segments_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(IsotonicRegressionTest, IsotonicRegression_test_case_uint32){
  ge::op::IsotonicRegression op;
  op.UpdateInputDesc("input", create_desc({4, 3, 4}, ge::DT_UINT32));
  op.SetAttr("output_dtype", ge::DT_DOUBLE);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("output");
  auto segments_desc = op.GetOutputDesc("segments");
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(segments_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_DOUBLE);
  EXPECT_EQ(segments_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(IsotonicRegressionTest, IsotonicRegression_test_case_int32){
  ge::op::IsotonicRegression op;
  op.UpdateInputDesc("input", create_desc({4, 3, 4}, ge::DT_INT32));
  op.SetAttr("output_dtype", ge::DT_DOUBLE);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("output");
  auto segments_desc = op.GetOutputDesc("segments");
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(segments_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_DOUBLE);
  EXPECT_EQ(segments_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(IsotonicRegressionTest, IsotonicRegression_test_case_uint64){
  ge::op::IsotonicRegression op;
  op.UpdateInputDesc("input", create_desc({4, 3, 4}, ge::DT_UINT64));
  op.SetAttr("output_dtype", ge::DT_DOUBLE);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("output");
  auto segments_desc = op.GetOutputDesc("segments");
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(segments_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_DOUBLE);
  EXPECT_EQ(segments_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(IsotonicRegressionTest, IsotonicRegression_test_case_int64){
  ge::op::IsotonicRegression op;
  op.UpdateInputDesc("input", create_desc({4, 3, 4}, ge::DT_INT64));
  op.SetAttr("output_dtype", ge::DT_DOUBLE);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("output");
  auto segments_desc = op.GetOutputDesc("segments");
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(segments_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_DOUBLE);
  EXPECT_EQ(segments_desc.GetDataType(), ge::DT_INT32);
}

TEST_F(IsotonicRegressionTest, IsotonicRegression_test_case_get_output_dtype_fail){
  ge::op::IsotonicRegression op;
  op.UpdateInputDesc("input", create_desc({4, 3, 4}, ge::DT_INT64));
  op.SetAttr("output_dtype", ge::DT_UINT16);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}