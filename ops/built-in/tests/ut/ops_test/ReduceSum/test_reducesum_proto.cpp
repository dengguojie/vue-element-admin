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
 * @file test_reducesum_d_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class ReduceSum : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ReduceSum SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ReduceSum TearDown" << std::endl;
  }
};

TEST_F(ReduceSum, reducesum_infer_shape_001) {
  ge::op::ReduceSum op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({-1, 100, 4},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);

  auto axes_desc = create_desc_shape_range({2,},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {2,},
                                           ge::FORMAT_ND, {{2, 2},});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("axes", axes_desc);
  op.SetAttr("keep_dims", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
    {1, 2},
    {1, 200},
    {1, 8},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ReduceSum, reducesum_infer_shape_002) {
  ge::op::ReduceSum op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({-1, 100, 4},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);
  auto axes_desc = create_desc_shape_range({2,},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {2,},
                                           ge::FORMAT_ND, {{2, 2},});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("axes", axes_desc);
  op.SetAttr("keep_dims", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, };
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
    {2, 200}
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ReduceSum, reducesum_infer_shape_003) {
  ge::op::ReduceSum op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {7, 7}, {3, 3}};
  auto tensor_desc = create_desc_shape_range({2, 7, 3},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 7, 3},
                                             ge::FORMAT_ND, shape_range);

  auto axes_desc = create_desc_shape_range({-2,},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {-2,},
                                           ge::FORMAT_ND, {});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("axes", axes_desc);
  op.SetAttr("keep_dims", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
    {1, 2},
    {1, 7},
    {1, 3},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ReduceSum, reducesum_infer_shape_004) {
  ge::op::ReduceSum op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 25}, {1, 25}, {1, 25}, {3, 3}, {1, 1}};
  auto tensor_desc = create_desc_shape_range({-1, -1, -1, 3, 1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {-1, -1, -1, 3, 1},
                                             ge::FORMAT_ND, shape_range);

  auto axes_desc = create_desc_shape_range({2,},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {2,},
                                           ge::FORMAT_ND, {});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("axes", axes_desc);
  op.SetAttr("keep_dims", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
    {1, 25},
    {1, 25},
    {1, 25},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ReduceSum, reducesum_infer_shape_005) {
  ge::op::ReduceSum op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {13, 13}, {13, 13}, {3, 3}, {1, 1}};
  auto tensor_desc = create_desc_shape_range({2, 13, 13, 3, 1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 13, 13, 3, 1},
                                             ge::FORMAT_ND, shape_range);

  auto axes_desc = create_desc_shape_range({-1,},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {-1,},
                                           ge::FORMAT_ND, {});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("axes", axes_desc);
  op.SetAttr("keep_dims", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
    {1, 2},
    {1, 13},
    {1, 13},
    {1, 3},
    {1, 1},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ReduceSum, reducesum_infer_shape_006) {
  ge::op::ReduceSum op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {7, 7}, {3, 3}};
  auto tensor_desc = create_desc_shape_range({2, 7, 3},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 7, 3},
                                             ge::FORMAT_ND, shape_range);

  auto axes_desc = create_desc_shape_range({-2,},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {-2,},
                                           ge::FORMAT_ND, {});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("axes", axes_desc);
  op.SetAttr("keep_dims", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ReduceSum, reducesum_infer_shape_007) {
  ge::op::ReduceSum op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {4, 7}, {3, 3}};
  auto tensor_desc = create_desc_shape_range({2, -1, 3},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, -1, 3},
                                             ge::FORMAT_ND, shape_range);

  auto axes_desc = create_desc_shape_range({-1,},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {-1,},
                                           ge::FORMAT_ND, {});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("axes", axes_desc);
  op.SetAttr("keep_dims", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
    {1, 2},
    {1, 7},
    {1, 3},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ReduceSum, reducesum_infer_shape_008) {
  ge::op::ReduceSum op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);

  auto axes_desc = create_desc_shape_range({},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {},
                                           ge::FORMAT_ND, {});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("axes", axes_desc);
  op.SetAttr("keep_dims", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ReduceSum, reducesum_infer_shape_009) {
  ge::op::ReduceSum op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);

  auto axes_desc = create_desc_shape_range({},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {},
                                           ge::FORMAT_ND, {});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("axes", axes_desc);
  op.SetAttr("keep_dims", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ReduceSum, reducesum_infer_shape_010) {
  ge::op::ReduceSum op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);

  auto axes_desc = create_desc_shape_range({-2,},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {-2,},
                                           ge::FORMAT_ND, {});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("axes", axes_desc);
  op.SetAttr("keep_dims", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ReduceSum, reducesum_infer_shape_011) {
  ge::op::ReduceSum op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);

  auto axes_desc = create_desc_shape_range({-2,},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {-2,},
                                           ge::FORMAT_ND, {});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("axes", axes_desc);
  op.SetAttr("keep_dims", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ReduceSum, reducesum_infer_shape_012) {
  ge::op::ReduceSum op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 25}, {2, 25}, {2, 25}, {3, 3}, {1, 1}};
  auto tensor_desc = create_desc_shape_range({-1, -1, -1, 3, 1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {-1, -1, -1, 3, 1},
                                             ge::FORMAT_ND, shape_range);

  auto axes_desc = create_desc_shape_range({1,},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {1,},
                                           ge::FORMAT_ND, {});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("axes", axes_desc);
  op.SetAttr("keep_dims", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1,-1,-1,-1,1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{1, 25}, {1, 25}, {1, 25}, {1, 3}, {1, 1}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ReduceSum, reducesum_infer_shape_013) {
  ge::op::ReduceSum op;
  std::vector<int64_t> dims_x{1024, 4};
  std::vector<int64_t> dims_axes{5,};
  ge::Shape shape_x(dims_x);
  ge::Shape shape_axes(dims_axes);
  ge::TensorDesc tensorDescX(shape_x, ge::FORMAT_ND, ge::DT_FLOAT);
  tensorDescX.SetOriginShape(shape_x);
  ge::TensorDesc tensorDescAxes(shape_axes, ge::FORMAT_ND, ge::DT_INT32);
  tensorDescAxes.SetOriginShape(shape_axes);

  op.UpdateInputDesc("x", tensorDescX);
  op.UpdateInputDesc("axes", tensorDescAxes);
  op.SetAttr("keep_dims", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ReduceSum, reducesum_infer_shape_014) {
  ge::op::ReduceSum op;
  std::vector<int64_t> dims_x{2,3};
  std::vector<int64_t> dims_axes{0,};
  ge::Shape shape_x(dims_x);
  ge::Shape shape_axes(dims_axes);
  ge::TensorDesc tensorDescX(shape_x, ge::FORMAT_ND, ge::DT_FLOAT);
  tensorDescX.SetOriginShape(shape_x);
  ge::TensorDesc tensorDescAxes(shape_axes, ge::FORMAT_ND, ge::DT_INT32);
  tensorDescAxes.SetOriginShape(shape_axes);

  op.UpdateInputDesc("x", tensorDescX);
  op.UpdateInputDesc("axes", tensorDescAxes);
  op.SetAttr("keep_dims", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {2,3};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}