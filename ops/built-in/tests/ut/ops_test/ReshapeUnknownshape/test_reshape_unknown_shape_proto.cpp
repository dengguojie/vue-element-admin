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
 * @file test_reshape_unknown_shape_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include <climits>
#include "op_proto_test_util.h"
#include "util.h"
#include "graph/utils/op_desc_utils.h"
#include "array_ops.h"
#include "graph/ge_tensor.h"
#include "graph/utils/graph_utils.h"

using std::make_pair;
static const int64_t UNKNOWN_DIM = -1;
class RESHAPE_UNKNOWN_SHAPE_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RESHAPE_UNKNOWN_SHAPE_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RESHAPE_UNKNOWN_SHAPE_UT TearDown" << std::endl;
  }
};

TEST_F(RESHAPE_UNKNOWN_SHAPE_UT, InferShape) {
  ge::op::Reshape op("Reshape");
  op.UpdateInputDesc("x", create_desc({-1}, ge::DT_INT32));
  op.UpdateInputDesc("shape", create_desc({}, ge::DT_INT32));
  op.SetAttr("axis", 0);
  op.SetAttr("num_axes", -1);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape output_shape;
  bool ret = SetScalarOutputDesc(std::string("x"), std::string("y"), op_desc, output_shape);

  int64_t a = 100;
  int64_t b = 10;
  ret = ge::array_ops::CheckInt64MulOverflow(a, b);

  a = 100;
  b = -10;
  ret = ge::array_ops::CheckInt64MulOverflow(a, b);

  a = -100;
  b = 10;
  ret = ge::array_ops::CheckInt64MulOverflow(a, b);

  a = -100;
  b = -10;
  ret = ge::array_ops::CheckInt64MulOverflow(a, b);
  EXPECT_EQ(ret, true);

  std::vector<std::pair<int64_t, int64_t>> x_range3;
  std::vector<std::pair<int64_t, int64_t>> y_range3;
  std::pair<int64_t,int64_t> pair3(16, 16);
  std::pair<int64_t,int64_t> pair4(1, -1);
  x_range3.push_back(pair3);
  x_range3.push_back(pair4);
  std::vector<int64_t> dims1{4,-1};
  ge::GeShape shape3(dims1);
  ge::array_ops::ReshapeRangeInfer(op, x_range3, y_range3, shape3);
  EXPECT_EQ(y_range3.size(), dims1.size());
  EXPECT_EQ(y_range3[1].first, 0);
  EXPECT_EQ(y_range3[1].second, -1);
  
  std::vector<std::pair<int64_t, int64_t>> x_range4;
  std::vector<std::pair<int64_t, int64_t>> y_range4;
  std::pair<int64_t, int64_t> pair5(1, 4);
  std::pair<int64_t,int64_t> pair6(16, 16);
  x_range4.push_back(pair5);
  x_range4.push_back(pair6);
  std::vector<int64_t> dims2{-1, 16};
  ge::GeShape shape4(dims2);
  ge::array_ops::ReshapeRangeInfer(op, x_range4, y_range4, shape4);
  EXPECT_EQ(y_range4.size(), dims2.size());
  EXPECT_EQ(y_range4[0].first, 1);
  EXPECT_EQ(y_range4[0].second, 4);
}

TEST_F(RESHAPE_UNKNOWN_SHAPE_UT, reshape_range_infer_test_range_max_exceeding_int32max) {
  int64_t oversized_dim = static_cast<int64_t>(INT32_MAX) + 100;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {make_pair(200, 200),
                                                          make_pair(1, oversized_dim),
                                                          make_pair(oversized_dim, -1),
                                                          make_pair(1, -1)};
  ge::GeShape shape = ge::GeShape({200, -1, -1, -1});

  ge::array_ops::FixRangeMaxToInt32max(shape, shape_range);

  std::vector<int64_t> target_shape_dims = {200, -1, -1, -1};
  EXPECT_EQ(shape.GetDims(), target_shape_dims);
  std::vector<int64_t> target_shape_range = {200, 200, 1, INT32_MAX, INT32_MAX, -1, 1, -1};
  std::vector<int64_t> output_shape_range;
  for (auto pair : shape_range) {
    output_shape_range.push_back(pair.first);
    output_shape_range.push_back(pair.second);
  }
  EXPECT_EQ(output_shape_range, target_shape_range);
}

TEST_F(RESHAPE_UNKNOWN_SHAPE_UT, reshape_range_infer_test_precise_max) {
  ge::op::Reshape op("Reshape");
  auto rank = 4;
  std::vector<std::pair<int64_t, int64_t>> x_shape_range = {make_pair(1, 100), make_pair(1, 400)};
  ge::GeShape x_shape = ge::GeShape(std::vector<int64_t>(2, UNKNOWN_DIM));
  std::vector<std::pair<int64_t, int64_t>> shape_value_range = {make_pair(100, -1), make_pair(-1, -10),
                                                                make_pair(1, 20),   make_pair(10, 10)};
  std::vector<std::pair<int64_t, int64_t>> y_shape_range;
  ge::GeShape y_shape = ge::GeShape(std::vector<int64_t>(rank, UNKNOWN_DIM));
  ge::array_ops::ReshapeRangeInferAllDims(op, x_shape_range, x_shape, shape_value_range, rank, y_shape_range, y_shape);

  std::vector<int64_t> target_y_shape_dims = {-1, -1, -1, 10};
  EXPECT_EQ(y_shape.GetDims(), target_y_shape_dims);

  std::vector<int64_t> target_y_shape_range = {100, 4000, 1, 40, 1, 20, 10, 10};
  std::vector<int64_t> output_shape_range;
  for (auto pair : y_shape_range) {
    output_shape_range.push_back(pair.first);
    output_shape_range.push_back(pair.second);
  }
  EXPECT_EQ(output_shape_range, target_y_shape_range);
}

TEST_F(RESHAPE_UNKNOWN_SHAPE_UT, reshape_range_infer_test_unknown_max) {
  ge::op::Reshape op("Reshape");
  auto rank = 4;
  std::vector<std::pair<int64_t, int64_t>> x_shape_range = {make_pair(1, 100), make_pair(1, 400), make_pair(-1, -1)};
  ge::GeShape x_shape = ge::GeShape(std::vector<int64_t>(3, UNKNOWN_DIM));
  std::vector<std::pair<int64_t, int64_t>> shape_value_range = {make_pair(100, -1), make_pair(1, -10),
                                                                make_pair(1, 20),   make_pair(10, 10)};
  std::vector<std::pair<int64_t, int64_t>> y_shape_range;
  ge::GeShape y_shape = ge::GeShape(std::vector<int64_t>(rank, UNKNOWN_DIM));
  ge::array_ops::ReshapeRangeInferAllDims(op, x_shape_range, x_shape, shape_value_range, rank, y_shape_range, y_shape);

  std::vector<int64_t> target_y_shape_dims = {-1, -1, -1, 10};
  EXPECT_EQ(y_shape.GetDims(), target_y_shape_dims);

  std::vector<int64_t> target_y_shape_range = {100, -1, 1, -1, 1, 20, 10, 10};
  std::vector<int64_t> output_shape_range;
  for (auto pair : y_shape_range) {
    output_shape_range.push_back(pair.first);
    output_shape_range.push_back(pair.second);
  }
  EXPECT_EQ(output_shape_range, target_y_shape_range);
}

TEST_F(RESHAPE_UNKNOWN_SHAPE_UT, reshape_range_infer_test_lower_boundary) {
  ge::op::Reshape op("Reshape");
  auto rank = 4;
  std::vector<std::pair<int64_t, int64_t>> x_shape_range = {};
  ge::GeShape x_shape = ge::GeShape(std::vector<int64_t>(3, 100));
  std::vector<std::pair<int64_t, int64_t>> shape_value_range = {make_pair(100, -1), make_pair(1, -10),
                                                                make_pair(1, 20),   make_pair(10, 10)};
  std::vector<std::pair<int64_t, int64_t>> y_shape_range;
  ge::GeShape y_shape = ge::GeShape(std::vector<int64_t>(rank, UNKNOWN_DIM));
  ge::array_ops::ReshapeRangeInferAllDims(op, x_shape_range, x_shape, shape_value_range, rank, y_shape_range, y_shape);

  std::vector<int64_t> target_y_shape_dims = {-1, -1, -1, 10};
  EXPECT_EQ(y_shape.GetDims(), target_y_shape_dims);

  std::vector<int64_t> target_y_shape_range = {100, 100000, 1, 1000, 1, 20, 10, 10};
  std::vector<int64_t> output_shape_range;
  for (auto pair : y_shape_range) {
    output_shape_range.push_back(pair.first);
    output_shape_range.push_back(pair.second);
  }
  EXPECT_EQ(output_shape_range, target_y_shape_range);
}

TEST_F(RESHAPE_UNKNOWN_SHAPE_UT, reshape_range_infer_test_empty_tensor) {
  ge::op::Reshape op("Reshape");
  auto rank = 4;
  std::vector<std::pair<int64_t, int64_t>> x_shape_range = {make_pair(1, 100), make_pair(0, 0)};
  ge::GeShape x_shape = ge::GeShape({-1, 0});
  std::vector<std::pair<int64_t, int64_t>> shape_value_range = {make_pair(0, 0), make_pair(-1, -10),
                                                                make_pair(10, 20),   make_pair(100, 100)};
  std::vector<std::pair<int64_t, int64_t>> y_shape_range;
  ge::GeShape y_shape = ge::GeShape(std::vector<int64_t>(rank, UNKNOWN_DIM));
  ge::array_ops::ReshapeRangeInferAllDims(op, x_shape_range, x_shape, shape_value_range, rank, y_shape_range, y_shape);

  std::vector<int64_t> target_y_shape_dims = {0, -1, -1, 100};
  EXPECT_EQ(y_shape.GetDims(), target_y_shape_dims);

  std::vector<int64_t> target_y_shape_range = {0, 0, 1, -1, 10, 20, 100, 100};
  std::vector<int64_t> output_shape_range;
  for (auto pair : y_shape_range) {
    output_shape_range.push_back(pair.first);
    output_shape_range.push_back(pair.second);
  }
  EXPECT_EQ(output_shape_range, target_y_shape_range);
}


TEST_F(RESHAPE_UNKNOWN_SHAPE_UT, reshape_range_infer_test_probable_empty_tensor) {
  ge::op::Reshape op("Reshape");
  auto rank = 4;
  std::vector<std::pair<int64_t, int64_t>> x_shape_range = {make_pair(1, 100), make_pair(0, 100)};
  ge::GeShape x_shape = ge::GeShape({-1, -1});
  std::vector<std::pair<int64_t, int64_t>> shape_value_range = {make_pair(0, 5000), make_pair(-1, -10),
                                                                make_pair(10, 20),  make_pair(10, 10)};
  std::vector<std::pair<int64_t, int64_t>> y_shape_range;
  ge::GeShape y_shape = ge::GeShape(std::vector<int64_t>(rank, UNKNOWN_DIM));
  ge::array_ops::ReshapeRangeInferAllDims(op, x_shape_range, x_shape, shape_value_range, rank, y_shape_range, y_shape);

  std::vector<int64_t> target_y_shape_dims = {-1, -1, -1, 10};
  EXPECT_EQ(y_shape.GetDims(), target_y_shape_dims);

  std::vector<int64_t> target_y_shape_range = {0, 1000, 1, 1000, 10, 20, 10, 10};
  std::vector<int64_t> output_shape_range;
  for (auto pair : y_shape_range) {
    output_shape_range.push_back(pair.first);
    output_shape_range.push_back(pair.second);
  }
  EXPECT_EQ(output_shape_range, target_y_shape_range);
}

TEST_F(RESHAPE_UNKNOWN_SHAPE_UT, KnownShapeInferShape) {
  const std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {6, 6}};
  auto tensor_desc = create_desc_shape_range({2, 6}, ge::DT_INT32, ge::FORMAT_ND,
                                             {2, 6}, ge::FORMAT_ND, shape_range);
  auto data0 = ge::op::Data("data0").set_attr_index(0);
  data0.update_input_desc_x(tensor_desc);
  data0.update_output_desc_y(tensor_desc);

  ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[2] = {3, 4};
  auto const_op = ge::op::Const("Const").set_attr_value(
          ge::Tensor(const_desc, (uint8_t *)const_value, 2 * sizeof(int32_t)));

  ge::op::Reshape reshape_op("Reshape");
  reshape_op.UpdateInputDesc("x", create_desc({2, 6}, ge::DT_INT32));
  reshape_op.UpdateInputDesc("shape", create_desc({2}, ge::DT_INT32));
  reshape_op.update_output_desc_y(tensor_desc);
  reshape_op.SetAttr("axis", 0);
  reshape_op.SetAttr("num_axes", -1);
  reshape_op.set_input_x(data0).set_input_shape(const_op);

  std::vector<ge::Operator> inputs{data0};
  std::vector<ge::Operator> outputs{reshape_op};
  ge::Graph graph("reshape_infer_shape_test");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);

  auto node = compute_graph->FindFirstNodeMatchType("Reshape");
  EXPECT_NE(node, nullptr);

  ge::GeTensorDesc ge_tensor(ge::GeShape({2}), ge::FORMAT_ND, ge::DT_INT32);
  ge_tensor.SetOriginShape(ge::GeShape({2}));
  node->GetOpDesc()->UpdateInputDesc("shape", ge_tensor);
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);

  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);

  ge::TensorDesc td_y = op.GetOutputDescByName("y");
  EXPECT_EQ(td_y.GetDataType(), ge::DT_INT32);
  auto dims = td_y.GetShape().GetDims();
  EXPECT_EQ(dims.size(), 2);
  EXPECT_EQ(dims[0], 3);
  EXPECT_EQ(dims[1], 4);
}

TEST_F(RESHAPE_UNKNOWN_SHAPE_UT, UnKnownShapeInferShape) {
  auto tensor_desc = create_desc({-1, 4, 2, 256}, ge::DT_INT32);
  auto data0 = ge::op::Data("data0").set_attr_index(0);
  data0.update_input_desc_x(tensor_desc);
  data0.update_output_desc_y(tensor_desc);

  ge::TensorDesc const_desc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[3] = {0, 0, -1};
  auto const_op = ge::op::Const("Const").set_attr_value(
          ge::Tensor(const_desc, (uint8_t *)const_value, 3 * sizeof(int32_t)));

  ge::op::Reshape reshape_op("Reshape");
  reshape_op.UpdateInputDesc("x", create_desc({-1, 4, 2, 256}, ge::DT_INT32));
  reshape_op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT32));
  reshape_op.set_input_x(data0).set_input_shape(const_op);

  std::vector<ge::Operator> inputs{data0};
  std::vector<ge::Operator> outputs{reshape_op};
  ge::Graph graph("reshape_infer_shape_test");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  auto node = compute_graph->FindFirstNodeMatchType("Reshape");
  EXPECT_NE(node, nullptr);

  ge::GeTensorDesc ge_tensor(ge::GeShape({3}), ge::FORMAT_ND, ge::DT_INT32);
  ge_tensor.SetOriginShape(ge::GeShape({3}));
  node->GetOpDesc()->UpdateInputDesc("shape", ge_tensor);
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);
  op.SetAttr("allowzero", 0);
  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);
}

TEST_F(RESHAPE_UNKNOWN_SHAPE_UT, Reshape_infer_format_5D) {
  ge::TensorDesc desc0(ge::Shape({3, 2, 2, 256}), ge::FORMAT_NDHWC, ge::DT_INT32);
  ge::TensorDesc desc1(ge::Shape({5}), ge::FORMAT_NDHWC, ge::DT_INT32);
  ge::op::Reshape op("Reshape");
  op.UpdateInputDesc("x", desc0);
  op.UpdateInputDesc("shape", desc1);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_EQ(op_desc->CallInferFormatFunc(op), ge::GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->MutableInputDesc(0)->GetFormat(), ge::FORMAT_ND);
  ASSERT_EQ(op_desc->MutableInputDesc(0)->GetOriginFormat(), ge::FORMAT_ND);
  ASSERT_EQ(op_desc->MutableInputDesc(1)->GetFormat(), ge::FORMAT_ND);
  ASSERT_EQ(op_desc->MutableInputDesc(1)->GetOriginFormat(), ge::FORMAT_ND);
  ASSERT_EQ(op_desc->MutableOutputDesc(0)->GetFormat(), ge::FORMAT_ND);
  ASSERT_EQ(op_desc->MutableOutputDesc(0)->GetOriginFormat(), ge::FORMAT_ND);
}
