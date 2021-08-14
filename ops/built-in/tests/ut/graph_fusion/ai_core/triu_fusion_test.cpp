#include <iostream>

#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include "matrix_calculation_ops.h"
#include "nn_pooling_ops.h"
#include "nonlinear_fuc_ops.h"

using namespace ge;
using namespace op;

class triu_fusion_test : public testing::Test {
  protected:
  static void SetUpTestCase() { std::cout << "triu_fusion SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "triu_fusion TearDown" << std::endl; }
};

TEST_F(triu_fusion_test, triu_fusion_test_000) {
  ge::Graph graph("triu_fusion_test_000");
  auto triu_input_data = op::Data("triu_input_data");
  std::vector<int64_t> dims{2, 16, 15};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
  triu_input_data.update_input_desc_x(tensorDesc);
  triu_input_data.update_output_desc_y(tensorDesc);
  auto triu_op = op::Triu("triu_0");
  triu_op.set_input_x(triu_input_data);
  triu_op.set_attr_diagonal(0);

  std::vector<Operator> inputs{triu_input_data};
  std::vector<Operator> outputs{triu_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TriuFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool triu_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      triu_match = true;
    }
  }
  EXPECT_EQ(triu_match, true);
}

TEST_F(triu_fusion_test, triu_fusion_test_001) {
  ge::Graph graph("triu_fusion_test_001");
  auto triu_input_data = op::Data("triu_input_data");
  std::vector<int64_t> dims{2, 16, 15};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
  triu_input_data.update_input_desc_x(tensorDesc);
  triu_input_data.update_output_desc_y(tensorDesc);
  auto triu_op = op::Triu("triu_1");
  triu_op.set_input_x(triu_input_data);
  triu_op.set_attr_diagonal(1);

  std::vector<Operator> inputs{triu_input_data};
  std::vector<Operator> outputs{triu_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TriuFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool triu_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      triu_match = true;
    }
  }
  EXPECT_EQ(triu_match, true);
}

TEST_F(triu_fusion_test, triu_fusion_test_002) {
  ge::Graph graph("triu_fusion_test_002");
  auto triu_input_data = op::Data("triu_input_data");
  std::vector<int64_t> dims{2, 7, 16};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
  triu_input_data.update_input_desc_x(tensorDesc);
  triu_input_data.update_output_desc_y(tensorDesc);
  auto triu_op = op::Triu("triu_2");
  triu_op.set_input_x(triu_input_data);
  triu_op.set_attr_diagonal(-1);

  std::vector<Operator> inputs{triu_input_data};
  std::vector<Operator> outputs{triu_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TriuFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool triu_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      triu_match = true;
    }
  }
  EXPECT_EQ(triu_match, true);
}

TEST_F(triu_fusion_test, triu_fusion_test_003) {
  ge::Graph graph("triu_fusion_test_003");
  auto triu_input_data = op::Data("triu_input_data");
  std::vector<int64_t> dims{2, 15, 16};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
  triu_input_data.update_input_desc_x(tensorDesc);
  triu_input_data.update_output_desc_y(tensorDesc);
  auto triu_op = op::Triu("triu_3");
  triu_op.set_input_x(triu_input_data);
  triu_op.set_attr_diagonal(1);

  std::vector<Operator> inputs{triu_input_data};
  std::vector<Operator> outputs{triu_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TriuFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool triu_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      triu_match = true;
    }
  }
  EXPECT_EQ(triu_match, true);
}

TEST_F(triu_fusion_test, triu_fusion_test_004) {
  ge::Graph graph("triu_fusion_test_004");
  auto triu_input_data = op::Data("triu_input_data");
  std::vector<int64_t> dims{2, 16, 15};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT16);
  triu_input_data.update_input_desc_x(tensorDesc);
  triu_input_data.update_output_desc_y(tensorDesc);
  auto triu_op = op::Triu("triu_4");
  triu_op.set_input_x(triu_input_data);
  triu_op.set_attr_diagonal(0);

  std::vector<Operator> inputs{triu_input_data};
  std::vector<Operator> outputs{triu_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TriuFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool triu_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      triu_match = true;
    }
  }
  EXPECT_EQ(triu_match, true);
}

TEST_F(triu_fusion_test, triu_fusion_test_005) {
  ge::Graph graph("triu_fusion_test_005");
  auto triu_input_data = op::Data("triu_input_data");
  std::vector<int64_t> dims{2, 16, 15};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
  triu_input_data.update_input_desc_x(tensorDesc);
  triu_input_data.update_output_desc_y(tensorDesc);
  auto triu_op = op::Triu("triu_5");
  triu_op.set_input_x(triu_input_data);
  triu_op.set_attr_diagonal(0);

  std::vector<Operator> inputs{triu_input_data};
  std::vector<Operator> outputs{triu_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TriuFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool triu_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      triu_match = true;
    }
  }
  EXPECT_EQ(triu_match, true);
}

TEST_F(triu_fusion_test, triu_fusion_test_006) {
  ge::Graph graph("triu_fusion_test_006");
  auto triu_input_data = op::Data("triu_input_data");
  std::vector<int64_t> dims{2, 16, 15};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT8);
  triu_input_data.update_input_desc_x(tensorDesc);
  triu_input_data.update_output_desc_y(tensorDesc);
  auto triu_op = op::Triu("triu_6");
  triu_op.set_input_x(triu_input_data);
  triu_op.set_attr_diagonal(0);

  std::vector<Operator> inputs{triu_input_data};
  std::vector<Operator> outputs{triu_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TriuFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool triu_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      triu_match = true;
    }
  }
  EXPECT_EQ(triu_match, true);
}

TEST_F(triu_fusion_test, triu_fusion_test_007) {
  ge::Graph graph("triu_fusion_test_007");
  auto triu_input_data = op::Data("triu_input_data");
  std::vector<int64_t> dims{2, 16, 15};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_UINT8);
  triu_input_data.update_input_desc_x(tensorDesc);
  triu_input_data.update_output_desc_y(tensorDesc);
  auto triu_op = op::Triu("triu_7");
  triu_op.set_input_x(triu_input_data);
  triu_op.set_attr_diagonal(0);

  std::vector<Operator> inputs{triu_input_data};
  std::vector<Operator> outputs{triu_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TriuFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool triu_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      triu_match = true;
    }
  }
  EXPECT_EQ(triu_match, true);
}