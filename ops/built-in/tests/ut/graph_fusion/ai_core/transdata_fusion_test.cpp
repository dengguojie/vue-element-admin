/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class transdata_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "transdata_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "transdata_fusion_test TearDown" << std::endl;
  }
};

TEST_F(transdata_fusion_test, transdata_fusion_test_1) {
  ge::Graph graph("transdata_fusion_test_1");
  auto input = op::Data().set_attr_index(0);
  auto trans = op::TransData("transdata").set_input_src(input);

  std::vector<int64_t> input_vec{16, 2, 1, 1, 16};
  ge::Shape input_shape(input_vec);
  ge::TensorDesc input_desc(input_shape, FORMAT_NC1HWC0, DT_FLOAT16);
  std::vector<int64_t> output_vec{16, 1, 1, 32};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);
  trans.update_input_desc_src(input_desc);
  trans.update_output_desc_dst(output_desc);

  auto invert = op::Invert("invert").set_input_x(trans);
  invert.update_input_desc_x(output_desc);
  invert.update_output_desc_y(output_desc);

  std::vector<Operator> inputs{input};
  std::vector<Operator> outputs{invert};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransDataPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = true;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransData") {
      findOp = false;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(transdata_fusion_test, transdata_fusion_test_2) {
  ge::Graph graph("transdata_fusion_test_2");
  auto input = op::Data().set_attr_index(0);
  auto trans = op::TransData("transdata").set_input_src(input);

  std::vector<int64_t> input_vec{16, 2, 1, 1, 16};
  ge::Shape input_shape(input_vec);
  ge::TensorDesc input_desc(input_shape, FORMAT_NC1HWC0, DT_FLOAT16);
  std::vector<int64_t> output_vec{16, 32, 1, 1};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NCHW, DT_FLOAT16);
  trans.update_input_desc_src(input_desc);
  trans.update_output_desc_dst(output_desc);

  auto invert = op::Invert("invert").set_input_x(trans);
  invert.update_input_desc_x(output_desc);
  invert.update_output_desc_y(output_desc);

  std::vector<Operator> inputs{input};
  std::vector<Operator> outputs{invert};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransDataPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = true;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransData") {
      findOp = false;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(transdata_fusion_test, transdata_fusion_test_3) {
  ge::Graph graph("transdata_fusion_test_3");
  auto input = op::Data().set_attr_index(0);
  auto trans = op::TransData("transdata").set_input_src(input);

  std::vector<int64_t> input_vec{16, 2, 1, 1, 16};
  ge::Shape input_shape(input_vec);
  ge::TensorDesc input_desc(input_shape, FORMAT_NC1HWC0, DT_FLOAT16);
  std::vector<int64_t> output_vec{16, 1, 1, 32};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);
  trans.update_input_desc_src(output_desc);
  trans.update_output_desc_dst(input_desc);

  auto invert = op::Invert("invert").set_input_x(trans);
  invert.update_input_desc_x(input_desc);
  invert.update_output_desc_y(input_desc);

  std::vector<Operator> inputs{input};
  std::vector<Operator> outputs{invert};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransDataPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = true;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransData") {
      findOp = false;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(transdata_fusion_test, transdata_fusion_test_4) {
  ge::Graph graph("transdata_fusion_test_4");
  auto input = op::Data().set_attr_index(0);
  auto trans = op::TransData("transdata").set_input_src(input);

  std::vector<int64_t> input_vec{16, 2, 1, 1, 16};
  ge::Shape input_shape(input_vec);
  ge::TensorDesc input_desc(input_shape, FORMAT_NC1HWC0, DT_FLOAT16);
  std::vector<int64_t> output_vec{16, 32, 1, 1};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NCHW, DT_FLOAT16);
  trans.update_input_desc_src(output_desc);
  trans.update_output_desc_dst(input_desc);

  auto invert = op::Invert("invert").set_input_x(trans);
  invert.update_input_desc_x(input_desc);
  invert.update_output_desc_y(input_desc);

  std::vector<Operator> inputs{input};
  std::vector<Operator> outputs{invert};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransDataPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = true;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransData") {
      findOp = false;
    }
  }
  EXPECT_EQ(findOp, true);
}
TEST_F(transdata_fusion_test, transdata_fusion_test_5) {
  ge::Graph graph("transdata_fusion_test_5");
  auto input = op::Data().set_attr_index(0);
  auto trans = op::TransData("transdata").set_input_src(input);

  std::vector<int64_t> input_vec{16, 2, 2, 2, 16};
  ge::Shape input_shape(input_vec);
  ge::TensorDesc input_desc(input_shape, FORMAT_NC1HWC0, DT_FLOAT16);
  std::vector<int64_t> output_vec{16, 2, 2, 32};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);
  trans.update_input_desc_src(input_desc);
  trans.update_output_desc_dst(output_desc);

  auto invert = op::Invert("invert").set_input_x(trans);
  invert.update_input_desc_x(output_desc);
  invert.update_output_desc_y(output_desc);

  std::vector<Operator> inputs{input};
  std::vector<Operator> outputs{invert};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransDataPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = true;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransData") {
      findOp = false;
    }
  }
  EXPECT_EQ(findOp, false);
}
TEST_F(transdata_fusion_test, transdata_fusion_test_6) {
  ge::Graph graph("transdata_fusion_test_6");
  auto input = op::Data().set_attr_index(0);
  auto trans = op::TransData("transdata").set_input_src(input);

  std::vector<int64_t> input_vec{16, 2, 1, 1, 16};
  ge::Shape input_shape(input_vec);
  ge::TensorDesc input_desc(input_shape, FORMAT_NC1HWC0, DT_FLOAT16);
  std::vector<int64_t> output_vec{16, 1, 1, 31};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);
  trans.update_input_desc_src(output_desc);
  trans.update_output_desc_dst(input_desc);

  auto invert = op::Invert("invert").set_input_x(trans);
  invert.update_input_desc_x(input_desc);
  invert.update_output_desc_y(input_desc);

  std::vector<Operator> inputs{input};
  std::vector<Operator> outputs{invert};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransDataPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = true;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransData") {
      findOp = false;
    }
  }
  EXPECT_EQ(findOp, false);
}
TEST_F(transdata_fusion_test, transdata_fusion_test_7) {
  ge::Graph graph("transdata_fusion_test_7");
  auto input = op::Data().set_attr_index(0);
  auto trans = op::TransData("transdata").set_input_src(input);

  std::vector<int64_t> input_vec{-1, 2, 1, 1, 16};
  ge::Shape input_shape(input_vec);
  ge::TensorDesc input_desc(input_shape, FORMAT_NC1HWC0, DT_FLOAT16);
  std::vector<int64_t> output_vec{-1, 1, 1, 32};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_NHWC, DT_FLOAT16);
  trans.update_input_desc_src(output_desc);
  trans.update_output_desc_dst(input_desc);

  auto invert = op::Invert("invert").set_input_x(trans);
  invert.update_input_desc_x(input_desc);
  invert.update_output_desc_y(input_desc);

  std::vector<Operator> inputs{input};
  std::vector<Operator> outputs{invert};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransDataPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = true;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransData") {
      findOp = false;
    }
  }
  EXPECT_EQ(findOp, false);
}
