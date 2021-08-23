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

class tril_fusion_test : public testing::Test {
  protected:
  static void SetUpTestCase() { std::cout << "tril_fusion SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "tril_fusion TearDown" << std::endl; }
};

TEST_F(tril_fusion_test, tril_fusion_test_000) {
  ge::Graph graph("tril_fusion_test_000");
  auto tril_input_data = op::Data("tril_input_data");
  std::vector<int64_t> dims{2, 16, 15};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
  tril_input_data.update_input_desc_x(tensorDesc);
  tril_input_data.update_output_desc_y(tensorDesc);
  auto tril_op = op::Tril("tril_0");
  tril_op.set_input_x(tril_input_data);
  tril_op.set_attr_diagonal(0);

  std::vector<Operator> inputs{tril_input_data};
  std::vector<Operator> outputs{tril_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TrilFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool tril_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      tril_match = true;
    }
  }
  EXPECT_EQ(tril_match, true);
}

TEST_F(tril_fusion_test, tril_fusion_test_001) {
  ge::Graph graph("tril_fusion_test_001");
  auto tril_input_data = op::Data("tril_input_data");
  std::vector<int64_t> dims{2, 16, 15};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
  tril_input_data.update_input_desc_x(tensorDesc);
  tril_input_data.update_output_desc_y(tensorDesc);
  auto tril_op = op::Tril("tril_1");
  tril_op.set_input_x(tril_input_data);
  tril_op.set_attr_diagonal(1);

  std::vector<Operator> inputs{tril_input_data};
  std::vector<Operator> outputs{tril_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TrilFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool tril_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      tril_match = true;
    }
  }
  EXPECT_EQ(tril_match, true);
}

TEST_F(tril_fusion_test, tril_fusion_test_002) {
  ge::Graph graph("tril_fusion_test_002");
  auto tril_input_data = op::Data("tril_input_data");
  std::vector<int64_t> dims{2, 7, 16};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
  tril_input_data.update_input_desc_x(tensorDesc);
  tril_input_data.update_output_desc_y(tensorDesc);
  auto tril_op = op::Tril("tril_2");
  tril_op.set_input_x(tril_input_data);
  tril_op.set_attr_diagonal(-1);

  std::vector<Operator> inputs{tril_input_data};
  std::vector<Operator> outputs{tril_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TrilFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool tril_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      tril_match = true;
    }
  }
  EXPECT_EQ(tril_match, true);
}

TEST_F(tril_fusion_test, tril_fusion_test_003) {
  ge::Graph graph("tril_fusion_test_003");
  auto tril_input_data = op::Data("tril_input_data");
  std::vector<int64_t> dims{2, 15, 16};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
  tril_input_data.update_input_desc_x(tensorDesc);
  tril_input_data.update_output_desc_y(tensorDesc);
  auto tril_op = op::Tril("tril_3");
  tril_op.set_input_x(tril_input_data);
  tril_op.set_attr_diagonal(1);

  std::vector<Operator> inputs{tril_input_data};
  std::vector<Operator> outputs{tril_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TrilFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool tril_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      tril_match = true;
    }
  }
  EXPECT_EQ(tril_match, true);
}

TEST_F(tril_fusion_test, tril_fusion_test_004) {
  ge::Graph graph("tril_fusion_test_004");
  auto tril_input_data = op::Data("tril_input_data");
  std::vector<int64_t> dims{2, 16, 15};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT16);
  tril_input_data.update_input_desc_x(tensorDesc);
  tril_input_data.update_output_desc_y(tensorDesc);
  auto tril_op = op::Tril("tril_4");
  tril_op.set_input_x(tril_input_data);
  tril_op.set_attr_diagonal(0);

  std::vector<Operator> inputs{tril_input_data};
  std::vector<Operator> outputs{tril_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TrilFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool tril_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      tril_match = true;
    }
  }
  EXPECT_EQ(tril_match, true);
}

TEST_F(tril_fusion_test, tril_fusion_test_005) {
  ge::Graph graph("tril_fusion_test_005");
  auto tril_input_data = op::Data("tril_input_data");
  std::vector<int64_t> dims{2, 16, 15};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
  tril_input_data.update_input_desc_x(tensorDesc);
  tril_input_data.update_output_desc_y(tensorDesc);
  auto tril_op = op::Tril("tril_5");
  tril_op.set_input_x(tril_input_data);
  tril_op.set_attr_diagonal(0);

  std::vector<Operator> inputs{tril_input_data};
  std::vector<Operator> outputs{tril_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TrilFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool tril_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      tril_match = true;
    }
  }
  EXPECT_EQ(tril_match, true);
}

TEST_F(tril_fusion_test, tril_fusion_test_006) {
  ge::Graph graph("tril_fusion_test_006");
  auto tril_input_data = op::Data("tril_input_data");
  std::vector<int64_t> dims{2, 16, 15};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT8);
  tril_input_data.update_input_desc_x(tensorDesc);
  tril_input_data.update_output_desc_y(tensorDesc);
  auto tril_op = op::Tril("tril_6");
  tril_op.set_input_x(tril_input_data);
  tril_op.set_attr_diagonal(0);

  std::vector<Operator> inputs{tril_input_data};
  std::vector<Operator> outputs{tril_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TrilFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool tril_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      tril_match = true;
    }
  }
  EXPECT_EQ(tril_match, true);
}

TEST_F(tril_fusion_test, tril_fusion_test_007) {
  ge::Graph graph("tril_fusion_test_007");
  auto tril_input_data = op::Data("tril_input_data");
  std::vector<int64_t> dims{2, 16, 15};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_UINT8);
  tril_input_data.update_input_desc_x(tensorDesc);
  tril_input_data.update_output_desc_y(tensorDesc);
  auto tril_op = op::Tril("tril_7");
  tril_op.set_input_x(tril_input_data);
  tril_op.set_attr_diagonal(0);

  std::vector<Operator> inputs{tril_input_data};
  std::vector<Operator> outputs{tril_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("TrilFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool tril_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      tril_match = true;
    }
  }
  EXPECT_EQ(tril_match, true);
}