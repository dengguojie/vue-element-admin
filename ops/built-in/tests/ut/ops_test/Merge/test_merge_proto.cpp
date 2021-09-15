/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_merge_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>

#include "control_flow_ops.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"

#include "op_proto_test_util.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"

class Merge : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Merge SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Merge TearDown" << std::endl;
  }
};

TEST_F(Merge, merge_infer_shape_known) {
  auto merge = ge::op::Merge("merge");
  EXPECT_EQ(merge.InferShapeAndType(), ge::GRAPH_FAILED); // in num invalid.

  const std::vector<std::pair<int64_t, int64_t>> shape_range = {};
  const auto tensor_desc = create_desc_shape_range({32},
                                                   ge::DT_FLOAT16, ge::FORMAT_ND,
                                                   {64},
                                                   ge::FORMAT_ND, shape_range);

  auto data0 = ge::op::Data("data0").set_attr_index(0);
  auto data1 = ge::op::Data("data1").set_attr_index(1);
  data0.update_input_desc_x(tensor_desc);
  data0.update_output_desc_y(tensor_desc);
  data1.update_input_desc_x(tensor_desc);
  data1.update_output_desc_y(tensor_desc);

  merge.create_dynamic_input_x(2);
  merge.UpdateDynamicInputDesc("x", 0, tensor_desc);
  merge.UpdateDynamicInputDesc("x", 1, tensor_desc);
  merge.set_dynamic_input_x(0, data0);
  merge.set_dynamic_input_x(1, data1);

  std::vector<ge::Operator> inputs{data0, data1};
  std::vector<ge::Operator> outputs{merge};
  ge::Graph graph("merge_infer_shape_test");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  auto node = compute_graph->FindFirstNodeMatchType("Merge");
  EXPECT_NE(node, nullptr);

  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);
  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);

  ge::TensorDesc td_y = op.GetOutputDescByName("y");
  std::vector<int64_t> expected_output_shape_y = {32};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range_y;
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range_y = {};
  EXPECT_EQ(td_y.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td_y.GetShape().GetDims(), expected_output_shape_y);
  EXPECT_EQ(td_y.GetShapeRange(output_shape_range_y), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape_range_y, expected_shape_range_y);

  ge::TensorDesc td_v = op.GetOutputDescByName("value_index");
  std::vector<int64_t> expected_output_shape_v = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range_v;
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range_v = {};
  EXPECT_EQ(td_v.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(td_v.GetShape().GetDims(), expected_output_shape_v);
  EXPECT_EQ(td_v.GetShapeRange(output_shape_range_v), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape_range_v, expected_shape_range_v);
}

TEST_F(Merge, merge_infer_shape_unknown) {
  const std::vector<std::pair<int64_t,int64_t>> shape_range = {{16, 128}};
  const auto tensor_desc = create_desc_shape_range({-1},
                                                   ge::DT_FLOAT16, ge::FORMAT_ND,
                                                   {64},
                                                   ge::FORMAT_ND, shape_range);

  auto data0 = ge::op::Data("data0").set_attr_index(0);
  auto data1 = ge::op::Data("data1").set_attr_index(1);
  auto data2 = ge::op::Data("data2").set_attr_index(2);
  data0.update_input_desc_x(tensor_desc);
  data0.update_output_desc_y(tensor_desc);
  data1.update_input_desc_x(tensor_desc);
  data1.update_output_desc_y(tensor_desc);
  data2.update_input_desc_x(tensor_desc);
  data2.update_output_desc_y(tensor_desc);

  auto merge = ge::op::Merge("merge");
  merge.create_dynamic_input_x(3);
  merge.UpdateDynamicInputDesc("x", 0, tensor_desc);
  merge.UpdateDynamicInputDesc("x", 1, tensor_desc);
  merge.UpdateDynamicInputDesc("x", 2, tensor_desc);
  merge.set_dynamic_input_x(0, data0);
  merge.set_dynamic_input_x(1, data1);
  merge.set_dynamic_input_x(2, data2);

  std::vector<ge::Operator> inputs{data0, data1, data2};
  std::vector<ge::Operator> outputs{merge};
  ge::Graph graph("merge_infer_shape_test");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  auto node = compute_graph->FindFirstNodeMatchType("Merge");
  EXPECT_NE(node, nullptr);

  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);
  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);

  ge::TensorDesc td_y = op.GetOutputDescByName("y");
  std::vector<int64_t> expected_output_shape_y = {-1};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range_y;
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range_y = { {16, 128} };
  EXPECT_EQ(td_y.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td_y.GetShape().GetDims(), expected_output_shape_y);
  EXPECT_EQ(td_y.GetShapeRange(output_shape_range_y), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape_range_y, expected_shape_range_y);

  ge::TensorDesc td_v = op.GetOutputDescByName("value_index");
  std::vector<int64_t> expected_output_shape_v = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range_v;
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range_v = {};
  EXPECT_EQ(td_v.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(td_v.GetShape().GetDims(), expected_output_shape_v);
  EXPECT_EQ(td_v.GetShapeRange(output_shape_range_v), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape_range_v, expected_shape_range_v);
}

TEST_F(Merge, merge_infer_running_calculation) {
  const std::vector<std::pair<int64_t,int64_t>> shape_range = {{32, 64}};
  const auto tensor_desc_x0 = create_desc_shape_range({32},
                                                      ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {64},
                                                      ge::FORMAT_ND, shape_range);
  const auto tensor_desc_x1 = create_desc_shape_range({64},
                                                      ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {64},
                                                      ge::FORMAT_ND, shape_range);

  auto data0 = ge::op::Data("data0").set_attr_index(0);
  auto data1 = ge::op::Data("data1").set_attr_index(1);
  data0.update_input_desc_x(tensor_desc_x0);
  data0.update_output_desc_y(tensor_desc_x0);
  data1.update_input_desc_x(tensor_desc_x1);
  data1.update_output_desc_y(tensor_desc_x1);

  auto merge = ge::op::Merge("merge");
  merge.create_dynamic_input_x(2);
  merge.UpdateDynamicInputDesc("x", 0, tensor_desc_x0);
  merge.UpdateDynamicInputDesc("x", 1, tensor_desc_x1);
  merge.set_dynamic_input_x(0, data0);
  merge.set_dynamic_input_x(1, data1);

  std::vector<ge::Operator> inputs{data0, data1};
  std::vector<ge::Operator> outputs{merge};
  ge::Graph graph("merge_infer_shape_test");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  auto node = compute_graph->FindFirstNodeMatchType("Merge");
  EXPECT_NE(node, nullptr);

  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);
  op.SetAttr("_merge_input_index", 3);
  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_FAILED);

  op.SetAttr("_merge_input_index", 1);
  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);

  ge::TensorDesc td_y = op.GetOutputDescByName("y");
  std::vector<int64_t> expected_output_shape_y = {64};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range_y;
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range_y = {};
  EXPECT_EQ(td_y.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td_y.GetShape().GetDims(), expected_output_shape_y);
  EXPECT_EQ(td_y.GetShapeRange(output_shape_range_y), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape_range_y, expected_shape_range_y);

  ge::TensorDesc td_v = op.GetOutputDescByName("value_index");
  std::vector<int64_t> expected_output_shape_v = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range_v;
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range_v = {};
  EXPECT_EQ(td_v.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(td_v.GetShape().GetDims(), expected_output_shape_v);
  EXPECT_EQ(td_v.GetShapeRange(output_shape_range_v), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape_range_v, expected_shape_range_v);
}

TEST_F(Merge, merge_infer_loopcond) {
  const std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  const auto tensor_desc_x0 = create_desc_shape_range({},
                                                      ge::DT_BOOL, ge::FORMAT_ND,
                                                      {},
                                                      ge::FORMAT_ND, shape_range);
  const auto tensor_desc_x1 = create_desc_shape_range({32},
                                                      ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {64},
                                                      ge::FORMAT_ND, shape_range);
  const auto tensor_desc_x2 = create_desc_shape_range({64},
                                                      ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {64},
                                                      ge::FORMAT_ND, shape_range);
  const auto tensor_desc_dummy = create_desc_shape_range({-3},
                                                        ge::DT_FLOAT16, ge::FORMAT_ND,
                                                        {},
                                                        ge::FORMAT_ND, shape_range);

  auto data0 = ge::op::Data("data0").set_attr_index(0);
  auto data1 = ge::op::Data("data1").set_attr_index(1);
  auto data2 = ge::op::Data("data2").set_attr_index(2);
  data0.update_input_desc_x(tensor_desc_x0);
  data0.update_output_desc_y(tensor_desc_x0);
  data1.update_input_desc_x(tensor_desc_x1);
  data1.update_output_desc_y(tensor_desc_x1);
  data2.update_input_desc_x(tensor_desc_x2);
  data2.update_output_desc_y(tensor_desc_x2);

  auto merge = ge::op::Merge("merge");
  merge.create_dynamic_input_x(2);
  merge.UpdateDynamicInputDesc("x", 0, tensor_desc_x1);
  merge.UpdateDynamicInputDesc("x", 1, tensor_desc_dummy);
  merge.set_dynamic_input_x(0, data1);

  auto switch0 = ge::op::Switch("switch");
  switch0.update_input_desc_data(tensor_desc_x1);
  switch0.update_input_desc_pred(tensor_desc_x0);
  switch0.update_output_desc_output_false(tensor_desc_x1);
  switch0.update_output_desc_output_true(tensor_desc_x1);
  switch0.set_input_data_by_name(merge, "y");
  switch0.set_input_pred(data0);

  auto exit1 = ge::op::Exit("exit");
  exit1.set_input_x_by_name(switch0, "output_false");

  auto add1 = ge::op::Add("add1");
  add1.set_input_x1(data2);
  add1.set_input_x2_by_name(switch0, "output_true");

  auto next1 = ge::op::NextIteration("next_iteration");
  next1.update_input_desc_x(tensor_desc_x2);
  next1.update_output_desc_y(tensor_desc_dummy);
  next1.set_input_x(add1);
  merge.set_dynamic_input_x(1, next1, "y");

  std::vector<ge::Operator> inputs{data0, data1, data2};
  std::vector<ge::Operator> outputs{exit1};
  ge::Graph graph("merge_infer_shape_test");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  auto node = compute_graph->FindFirstNodeMatchType("Merge");
  EXPECT_NE(node, nullptr);

  // Infer first time.
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);
  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);
  {
    ge::TensorDesc td_y = op.GetOutputDescByName("y");
    std::vector<int64_t> expected_output_shape_y = {32}; // task shape as Enter.
    std::vector<std::pair<int64_t, int64_t>> output_shape_range_y;
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range_y = {};
    EXPECT_EQ(td_y.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(td_y.GetShape().GetDims(), expected_output_shape_y);
    EXPECT_EQ(td_y.GetShapeRange(output_shape_range_y), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_shape_range_y, expected_shape_range_y);

    ge::TensorDesc td_v = op.GetOutputDescByName("value_index");
    std::vector<int64_t> expected_output_shape_v = {};
    std::vector<std::pair<int64_t, int64_t>> output_shape_range_v;
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range_v = {};
    EXPECT_EQ(td_v.GetDataType(), ge::DT_INT32);
    EXPECT_EQ(td_v.GetShape().GetDims(), expected_output_shape_v);
    EXPECT_EQ(td_v.GetShapeRange(output_shape_range_v), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_shape_range_v, expected_shape_range_v);
  }
  next1.update_output_desc_y(tensor_desc_x2);
  merge.set_dynamic_input_x(1, next1, "y");

  // Infer second time.
  op.SetAttr("_need_infer_again", true);
  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);
  {
    ge::TensorDesc td_y = op.GetOutputDescByName("y");
    std::vector<int64_t> expected_output_shape_y = {-1};
    std::vector<std::pair<int64_t, int64_t>> output_shape_range_y;
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range_y = {{32, 64}};
    EXPECT_EQ(td_y.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(td_y.GetShape().GetDims(), expected_output_shape_y);
    EXPECT_EQ(td_y.GetShapeRange(output_shape_range_y), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_shape_range_y, expected_shape_range_y);

    ge::TensorDesc td_v = op.GetOutputDescByName("value_index");
    std::vector<int64_t> expected_output_shape_v = {};
    std::vector<std::pair<int64_t, int64_t>> output_shape_range_v;
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range_v = {};
    EXPECT_EQ(td_v.GetDataType(), ge::DT_INT32);
    EXPECT_EQ(td_v.GetShape().GetDims(), expected_output_shape_v);
    EXPECT_EQ(td_v.GetShapeRange(output_shape_range_v), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_shape_range_v, expected_shape_range_v);
  }
}

TEST_F(Merge, merge_infer_normal_unknown) {
  const std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  const auto tensor_desc_x0 = create_desc_shape_range({32},
                                                      ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {64},
                                                      ge::FORMAT_ND, shape_range);
  const auto tensor_desc_x1 = create_desc_shape_range({64},
                                                      ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {64},
                                                      ge::FORMAT_ND, shape_range);
  auto data0 = ge::op::Data("data0").set_attr_index(0);
  auto data1 = ge::op::Data("data1").set_attr_index(1);
  data0.update_input_desc_x(tensor_desc_x0);
  data0.update_output_desc_y(tensor_desc_x0);
  data1.update_input_desc_x(tensor_desc_x1);
  data1.update_output_desc_y(tensor_desc_x1);

  auto merge = ge::op::Merge("merge");
  merge.create_dynamic_input_x(2);
  merge.UpdateDynamicInputDesc("x", 0, tensor_desc_x1);
  merge.UpdateDynamicInputDesc("x", 1, tensor_desc_x0);
  merge.set_dynamic_input_x(0, data1);
  merge.set_dynamic_input_x(1, data0);

  std::vector<ge::Operator> inputs{data0, data1};
  std::vector<ge::Operator> outputs{merge};
  ge::Graph graph("merge_infer_shape_test");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  auto node = compute_graph->FindFirstNodeMatchType("Merge");
  EXPECT_NE(node, nullptr);

  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);
  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);

  ge::TensorDesc td_y = op.GetOutputDescByName("y");
  std::vector<int64_t> expected_output_shape_y = {-1};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range_y;
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range_y = {{32, 64}};
  EXPECT_EQ(td_y.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td_y.GetShape().GetDims(), expected_output_shape_y);
  EXPECT_EQ(td_y.GetShapeRange(output_shape_range_y), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape_range_y, expected_shape_range_y);

  ge::TensorDesc td_v = op.GetOutputDescByName("value_index");
  std::vector<int64_t> expected_output_shape_v = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range_v;
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range_v = {};
  EXPECT_EQ(td_v.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(td_v.GetShape().GetDims(), expected_output_shape_v);
  EXPECT_EQ(td_v.GetShapeRange(output_shape_range_v), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape_range_v, expected_shape_range_v);
}

TEST_F(Merge, pass_throw_infer) {
  auto next_iteration = ge::op::NextIteration("next_iteration");
  auto ret = next_iteration.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// if infered shape same with pre out shape, but range not same
// set shape range to [1-1]
TEST_F(Merge, while_shape_range_protect) {
  // build graph 
  // data0   next_iteration(shape:[-1]; range[1,10])
  //    \   /
  //    merge(shape:[-1]; range[1,9])
  const std::vector<std::pair<int64_t,int64_t>> shape_range_x1 = {{1,10}};
  const std::vector<std::pair<int64_t,int64_t>> shape_range_y = {{1,9}};
  const auto tensor_desc_x0 = create_desc_shape_range({32},
                                                      ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {64},
                                                      ge::FORMAT_ND, {});
  const auto tensor_desc_x1 = create_desc_shape_range({-1},
                                                      ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {32},
                                                      ge::FORMAT_ND, shape_range_x1);
  const auto tensor_desc_y = create_desc_shape_range({-1},
                                                      ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {32},
                                                      ge::FORMAT_ND, shape_range_y);
  auto data0 = ge::op::Data("data0").set_attr_index(0);
  data0.update_input_desc_x(tensor_desc_x0);
  data0.update_output_desc_y(tensor_desc_x0);
  auto next1 = ge::op::NextIteration("next_iteration");
  next1.update_input_desc_x(tensor_desc_x1);
  next1.update_output_desc_y(tensor_desc_x1);

  auto merge = ge::op::Merge("merge");
  merge.create_dynamic_input_x(2);
  merge.UpdateDynamicInputDesc("x", 0, tensor_desc_x0);
  merge.UpdateDynamicInputDesc("x", 1, tensor_desc_x1);
  merge.update_output_desc_y(tensor_desc_y);
  merge.set_dynamic_input_x(0, data0);
  merge.set_dynamic_input_x(1, next1);

  std::vector<ge::Operator> inputs{data0, next1};
  std::vector<ge::Operator> outputs{merge};
  ge::Graph graph("merge_infer_shape_test");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  auto node = compute_graph->FindFirstNodeMatchType("Merge");
  EXPECT_NE(node, nullptr);

  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);
  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);

  ge::TensorDesc td_y = op.GetOutputDescByName("y");
  std::vector<int64_t> expected_output_shape_y = {-1};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range_y;
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range_y = {{1, -1}};
  EXPECT_EQ(td_y.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td_y.GetShape().GetDims(), expected_output_shape_y);
  EXPECT_EQ(td_y.GetShapeRange(output_shape_range_y), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape_range_y, expected_shape_range_y);
}

