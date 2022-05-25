/**
 * Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-1.0
 *
 * @file test_BandedTriangularSolve_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "linalg_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class bandedtriangularsolve_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "bandedtriangularsolve_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "bandedtriangularsolve_infer_test TearDown" << std::endl;
  }
};

TEST_F(bandedtriangularsolve_infer_test, bandedtriangularsolve_infer_test_1) {
  //new op
  ge::op::BandedTriangularSolve op;
  // set input info
  ge::TensorDesc tensor_desc_bands = create_desc_shape_range({2,4},ge::DT_DOUBLE, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("bands", tensor_desc_bands);
  ge::TensorDesc tensor_desc_rhs = create_desc_shape_range({4,1},ge::DT_DOUBLE,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("rhs", tensor_desc_rhs);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto states = op.VerifyAllAttr(true);
  EXPECT_EQ(states, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("output");
  std::vector<int64_t> expected_output_shape = {4,1};
  
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_DOUBLE);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);  
}

TEST_F(bandedtriangularsolve_infer_test, bandedtriangularsolve_infer_test_2) {
  //new op
  ge::op::BandedTriangularSolve op;
  // set input info
  ge::TensorDesc tensor_desc_bands = create_desc_shape_range({2,2,3},ge::DT_FLOAT, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("bands", tensor_desc_bands);
  ge::TensorDesc tensor_desc_rhs = create_desc_shape_range({2,3,1},ge::DT_FLOAT,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("rhs", tensor_desc_rhs);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto states = op.VerifyAllAttr(true);
  EXPECT_EQ(states, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("output");
  std::vector<int64_t> expected_output_shape = {2,3,1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
}

TEST_F(bandedtriangularsolve_infer_test, bandedtriangularsolve_infer_test_3) {
  //new op
  ge::op::BandedTriangularSolve op;
  // set input info
  ge::TensorDesc tensor_desc_bands = create_desc_shape_range({2,2,3},ge::DT_FLOAT16, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("bands", tensor_desc_bands);
  ge::TensorDesc tensor_desc_rhs = create_desc_shape_range({2,3,1},ge::DT_FLOAT16,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("rhs", tensor_desc_rhs);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto states = op.VerifyAllAttr(true);
  EXPECT_EQ(states, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("output");
  std::vector<int64_t> expected_output_shape = {2,3,1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

TEST_F(bandedtriangularsolve_infer_test, bandedtriangularsolve_infer_test_4) {
  //new op
  ge::op::BandedTriangularSolve op;
  // set input info
  ge::TensorDesc tensor_desc_bands = create_desc_shape_range({2,2,3},ge::DT_COMPLEX64, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("bands", tensor_desc_bands);
  ge::TensorDesc tensor_desc_rhs = create_desc_shape_range({2,3,1},ge::DT_COMPLEX64,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("rhs", tensor_desc_rhs);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto states = op.VerifyAllAttr(true);
  EXPECT_EQ(states, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("output");
  std::vector<int64_t> expected_output_shape = {2,3,1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_COMPLEX64);
}

TEST_F(bandedtriangularsolve_infer_test, bandedtriangularsolve_infer_test_5) {
  //new op
  ge::op::BandedTriangularSolve op;
  // set input info
  ge::TensorDesc tensor_desc_bands = create_desc_shape_range({2,2,3},ge::DT_COMPLEX128, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("bands", tensor_desc_bands);
  ge::TensorDesc tensor_desc_rhs = create_desc_shape_range({2,3,1},ge::DT_COMPLEX128,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("rhs", tensor_desc_rhs);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto states = op.VerifyAllAttr(true);
  EXPECT_EQ(states, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("output");
  std::vector<int64_t> expected_output_shape = {2,3,1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_COMPLEX128);
}

TEST_F(bandedtriangularsolve_infer_test, bandedtriangularsolve_infer_test_6) {
  //new op
  ge::op::BandedTriangularSolve op;
  // set input info
  ge::TensorDesc tensor_desc_bands = create_desc_shape_range({2,2,3},ge::DT_INT64, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("bands", tensor_desc_bands);
  ge::TensorDesc tensor_desc_rhs = create_desc_shape_range({2,3,1},ge::DT_COMPLEX128,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("rhs", tensor_desc_rhs);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto states = op.VerifyAllAttr(true);
  EXPECT_EQ(states, ge::GRAPH_FAILED);  
}

TEST_F(bandedtriangularsolve_infer_test, bandedtriangularsolve_infer_test_7) {
  //new op
  ge::op::BandedTriangularSolve op;
  // set input info
  ge::TensorDesc tensor_desc_bands = create_desc_shape_range({2,2,3},ge::DT_FLOAT, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("bands", tensor_desc_bands);
  ge::TensorDesc tensor_desc_rhs = create_desc_shape_range({2,4,1},ge::DT_FLOAT,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("rhs", tensor_desc_rhs);
  auto states = op.VerifyAllAttr(true);
  EXPECT_EQ(states, ge::GRAPH_FAILED);
}

TEST_F(bandedtriangularsolve_infer_test, bandedtriangularsolve_infer_test_8) {
  //new op
  ge::op::BandedTriangularSolve op;
  // set input info
  ge::TensorDesc tensor_desc_bands = create_desc_shape_range({2,2,2,3},ge::DT_FLOAT, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("bands", tensor_desc_bands);
  ge::TensorDesc tensor_desc_rhs = create_desc_shape_range({2,2,4,1},ge::DT_FLOAT,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("rhs", tensor_desc_rhs);
  auto states = op.VerifyAllAttr(true);
  EXPECT_EQ(states, ge::GRAPH_FAILED);
}

TEST_F(bandedtriangularsolve_infer_test, bandedtriangularsolve_infer_test_9) {
  //new op
  ge::op::BandedTriangularSolve op;
  // set input info
  ge::TensorDesc tensor_desc_bands = create_desc_shape_range({2,2,2,3},ge::DT_FLOAT, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("bands", tensor_desc_bands);
  ge::TensorDesc tensor_desc_rhs = create_desc_shape_range({2,3,3,1},ge::DT_FLOAT,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("rhs", tensor_desc_rhs);
  auto states = op.VerifyAllAttr(true);
  EXPECT_EQ(states, ge::GRAPH_FAILED);
}

TEST_F(bandedtriangularsolve_infer_test, bandedtriangularsolve_infer_test_10) {
  //new op
  ge::op::BandedTriangularSolve op;
  // set input info
  ge::TensorDesc tensor_desc_bands = create_desc_shape_range({2,2,2,3},ge::DT_FLOAT, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("bands", tensor_desc_bands);
  ge::TensorDesc tensor_desc_rhs = create_desc_shape_range({3,2,4,1},ge::DT_FLOAT,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("rhs", tensor_desc_rhs);
  auto states = op.VerifyAllAttr(true);
  EXPECT_EQ(states, ge::GRAPH_FAILED);
}

TEST_F(bandedtriangularsolve_infer_test, bandedtriangularsolve_infer_test_11) {
  //new op
  ge::op::BandedTriangularSolve op;
  // set input info
  ge::TensorDesc tensor_desc_bands = create_desc_shape_range({2,4},ge::DT_DOUBLE, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("bands", tensor_desc_bands);
  ge::TensorDesc tensor_desc_rhs = create_desc_shape_range({4,2},ge::DT_DOUBLE,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("rhs", tensor_desc_rhs);
  auto states = op.VerifyAllAttr(true);
  EXPECT_EQ(states, ge::GRAPH_FAILED);
}
