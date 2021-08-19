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
 * @file test_while_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>

#include "functional_ops.h"
#include "array_ops.h"

#include "op_proto_test_util.h"
#include "graph/utils/graph_utils.h"
#include "utils/op_desc_utils.h"

class While : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "While SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "While TearDown" << std::endl;
  }
};

TEST_F(While, while_infer_shape_first_time) {
  auto while_op = ge::op::While("while");

  const std::vector<std::pair<int64_t, int64_t>> shape_range = {};
  const auto tensor_desc = create_desc_shape_range({32},
                                                   ge::DT_FLOAT16, ge::FORMAT_ND,
                                                   {64},
                                                   ge::FORMAT_ND, shape_range);
  const auto tensor_desc_dummy = create_desc_shape_range({-3},
                                                        ge::DT_FLOAT16, ge::FORMAT_ND,
                                                        {},
                                                        ge::FORMAT_ND, shape_range);

  auto data0 = ge::op::Data("data0").set_attr_index(0);
  auto data1 = ge::op::Data("data1").set_attr_index(1);
  data0.update_input_desc_x(tensor_desc);
  data0.update_output_desc_y(tensor_desc);
  data1.update_input_desc_x(tensor_desc);
  data1.update_output_desc_y(tensor_desc);

  while_op.create_dynamic_input_input(2);
  while_op.UpdateDynamicInputDesc("input", 0, tensor_desc);
  while_op.UpdateDynamicInputDesc("input", 1, tensor_desc);
  while_op.set_dynamic_input_input(0, data0);
  while_op.set_dynamic_input_input(1, data1);
  while_op.create_dynamic_output_output(2);
  while_op.UpdateDynamicOutputDesc("output",0, tensor_desc_dummy);
  while_op.UpdateDynamicOutputDesc("output",1, tensor_desc_dummy);


  std::vector<ge::Operator> inputs{data0, data1};
  std::vector<ge::Operator> outputs{while_op};
  ge::Graph graph("while_infer_shape_test");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  auto node = compute_graph->FindFirstNodeMatchType("While");
  EXPECT_NE(node, nullptr);

  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);
  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);
  auto output_desc = while_op.GetDynamicOutputDesc("output",0);
  EXPECT_EQ(output_desc.GetShape().GetDims(), std::vector<int64_t>({-3}));
}


TEST_F(While, while_infer_shape_second_time_get_unknown_shape) {
  auto while_op = ge::op::While("while");

  const std::vector<std::pair<int64_t, int64_t>> shape_range = {};
  const auto tensor_desc = create_desc_shape_range({32},
                                                   ge::DT_FLOAT16, ge::FORMAT_ND,
                                                   {64},
                                                   ge::FORMAT_ND, shape_range);
  const auto tensor_desc_out = create_desc_shape_range({64},
                                                        ge::DT_FLOAT16, ge::FORMAT_ND,
                                                        {64},
                                                        ge::FORMAT_ND, shape_range);

  auto data0 = ge::op::Data("data0").set_attr_index(0);
  auto data1 = ge::op::Data("data1").set_attr_index(1);
  data0.update_input_desc_x(tensor_desc);
  data0.update_output_desc_y(tensor_desc);
  data1.update_input_desc_x(tensor_desc);
  data1.update_output_desc_y(tensor_desc);

  while_op.create_dynamic_input_input(2);
  while_op.UpdateDynamicInputDesc("input", 0, tensor_desc);
  while_op.UpdateDynamicInputDesc("input", 1, tensor_desc);
  while_op.set_dynamic_input_input(0, data0);
  while_op.set_dynamic_input_input(1, data1);
  while_op.create_dynamic_output_output(2);
  while_op.UpdateDynamicOutputDesc("output",0, tensor_desc_out);
  while_op.UpdateDynamicOutputDesc("output",1, tensor_desc_out);


  std::vector<ge::Operator> inputs{data0, data1};
  std::vector<ge::Operator> outputs{while_op};
  ge::Graph graph("while_infer_shape_test");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  auto node = compute_graph->FindFirstNodeMatchType("While");
  EXPECT_NE(node, nullptr);

  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);
  //EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_FAILED); // local ok, online fail
  
  EXPECT_EQ(op.InferShapeAndType(), 50331647); // GRAPH_NODE_NEED_REPASS
  auto input_desc = while_op.GetDynamicInputDesc("input",0);
  // Check input is refreshed by output is unknown
  EXPECT_EQ(input_desc.GetShape().GetDims(), std::vector<int64_t>({-1}));

}

TEST_F(While, while_infer_shape_second_time_get_unknown_rank) {
  auto while_op = ge::op::While("while");

  const std::vector<std::pair<int64_t, int64_t>> shape_range = {};
  const auto tensor_desc = create_desc_shape_range({32},
                                                   ge::DT_FLOAT16, ge::FORMAT_ND,
                                                   {64},
                                                   ge::FORMAT_ND, shape_range);
  const auto tensor_desc_out = create_desc_shape_range({64, 32},
                                                        ge::DT_FLOAT16, ge::FORMAT_ND,
                                                        {64, 64},
                                                        ge::FORMAT_ND, shape_range);

  auto data0 = ge::op::Data("data0").set_attr_index(0);
  auto data1 = ge::op::Data("data1").set_attr_index(1);
  data0.update_input_desc_x(tensor_desc);
  data0.update_output_desc_y(tensor_desc);
  data1.update_input_desc_x(tensor_desc);
  data1.update_output_desc_y(tensor_desc);

  while_op.create_dynamic_input_input(2);
  while_op.UpdateDynamicInputDesc("input", 0, tensor_desc);
  while_op.UpdateDynamicInputDesc("input", 1, tensor_desc);
  while_op.set_dynamic_input_input(0, data0);
  while_op.set_dynamic_input_input(1, data1);
  while_op.create_dynamic_output_output(2);
  while_op.UpdateDynamicOutputDesc("output",0, tensor_desc_out);
  while_op.UpdateDynamicOutputDesc("output",1, tensor_desc_out);

  std::vector<ge::Operator> inputs{data0, data1};
  std::vector<ge::Operator> outputs{while_op};
  ge::Graph graph("while_infer_shape_test");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  auto node = compute_graph->FindFirstNodeMatchType("While");
  EXPECT_NE(node, nullptr);

  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);
  
  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);
  auto input_desc = while_op.GetDynamicInputDesc("input",0);
  // Check input is refreshed by output is unknown
  EXPECT_EQ(input_desc.GetShape().GetDims(), std::vector<int64_t>({-2}));
}

TEST_F(While, while_infer_shape_in_num_not_equal_out_num_failed) {
  auto while_op = ge::op::While("while");

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

  while_op.create_dynamic_input_input(2);
  while_op.UpdateDynamicInputDesc("input", 0, tensor_desc);
  while_op.UpdateDynamicInputDesc("input", 1, tensor_desc);
  while_op.set_dynamic_input_input(0, data0);
  while_op.set_dynamic_input_input(1, data1);

  std::vector<ge::Operator> inputs{data0, data1};
  std::vector<ge::Operator> outputs{while_op};
  ge::Graph graph("while_infer_shape_test");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  auto node = compute_graph->FindFirstNodeMatchType("While");
  EXPECT_NE(node, nullptr);

  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);
  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_FAILED);
}

TEST_F(While, while_infer_shape_input_dtype_not_equal_output_dtype_failed) {
  auto while_op = ge::op::While("while");

  const std::vector<std::pair<int64_t, int64_t>> shape_range = {};
  const auto tensor_desc = create_desc_shape_range({32},
                                                   ge::DT_FLOAT16, ge::FORMAT_ND,
                                                   {64},
                                                   ge::FORMAT_ND, shape_range);
  const auto tensor_desc_out = create_desc_shape_range({64},
                                                        ge::DT_FLOAT, ge::FORMAT_ND,
                                                        {64},
                                                        ge::FORMAT_ND, shape_range);

  auto data0 = ge::op::Data("data0").set_attr_index(0);
  auto data1 = ge::op::Data("data1").set_attr_index(1);
  data0.update_input_desc_x(tensor_desc);
  data0.update_output_desc_y(tensor_desc);
  data1.update_input_desc_x(tensor_desc);
  data1.update_output_desc_y(tensor_desc);

  while_op.create_dynamic_input_input(2);
  while_op.UpdateDynamicInputDesc("input", 0, tensor_desc);
  while_op.UpdateDynamicInputDesc("input", 1, tensor_desc);
  while_op.set_dynamic_input_input(0, data0);
  while_op.set_dynamic_input_input(1, data1);
  while_op.create_dynamic_output_output(2);
  while_op.UpdateDynamicOutputDesc("output",0, tensor_desc_out);
  while_op.UpdateDynamicOutputDesc("output",1, tensor_desc_out);


  std::vector<ge::Operator> inputs{data0, data1};
  std::vector<ge::Operator> outputs{while_op};
  ge::Graph graph("while_infer_shape_test");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  auto node = compute_graph->FindFirstNodeMatchType("While");
  EXPECT_NE(node, nullptr);

  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);
  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_FAILED); 
}
