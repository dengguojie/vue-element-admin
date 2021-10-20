#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "matrix_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
using namespace op;

class matrix_diag_part_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "matrix_diag_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "matrix_diag_fusion_test TearDown" << std::endl;
  }
};

TEST_F(matrix_diag_part_fusion_pass_test, matrix_diag_part_fusion_pass_test_1) {
  ge::Graph graph("matrix_diag_part_fusion_pass_test_1");
  auto matrix_diag_input_data = op::Data("matrix_diag_input_data");
  std::vector<int64_t> dims{128,32,16,32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
  matrix_diag_input_data.update_input_desc_x(tensorDesc);
  matrix_diag_input_data.update_output_desc_y(tensorDesc);
  auto matrix_diag_part_op = op::MatrixDiagPart("matrix_diag_part0");
  matrix_diag_part_op.set_input_x(matrix_diag_input_data);

  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(matrix_diag_part_op);

  std::vector<Operator> inputs{matrix_diag_input_data};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatrixDiagPartFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool avgPoolMatch = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatrixDiagPartD") {
      avgPoolMatch = true;
    }
  }
  EXPECT_EQ(avgPoolMatch, true);
}

TEST_F(matrix_diag_part_fusion_pass_test, matrix_diag_part_fusion_pass_test_2) {
  ge::Graph graph("matrix_diag_part_fusion_pass_test_2");
  auto matrix_diag_input_data = op::Data("matrix_diag_input_data");
  std::vector<int64_t> dims{128};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_INT32);
  matrix_diag_input_data.update_input_desc_x(tensorDesc);
  matrix_diag_input_data.update_output_desc_y(tensorDesc);
  auto matrix_diag_part_op = op::MatrixDiagPart("matrix_diag_0");
  matrix_diag_part_op.set_input_x(matrix_diag_input_data);

  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(matrix_diag_part_op);
  std::vector<Operator> inputs{matrix_diag_input_data};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatrixDiagPartFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool avgPoolMatch = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatrixDiagPartD") {
      avgPoolMatch = true;
    }
  }
  EXPECT_EQ(avgPoolMatch, true);
}

TEST_F(matrix_diag_part_fusion_pass_test, matrix_diag_part_fusion_pass_test_3) {
  ge::Graph graph("matrix_diag_part_fusion_pass_test_3");
  auto matrix_diag_input_data = op::Data("matrix_diag_input_data");
  std::vector<int64_t> dims{128};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
  matrix_diag_input_data.update_input_desc_x(tensorDesc);
  matrix_diag_input_data.update_output_desc_y(tensorDesc);
  auto matrix_diag_part_op = op::MatrixDiagPart("matrix_diag_0");
  matrix_diag_part_op.set_input_x(matrix_diag_input_data);

  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(matrix_diag_part_op);
  std::vector<Operator> inputs{matrix_diag_input_data};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatrixDiagPartFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool avgPoolMatch = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatrixDiagPartD") {
      avgPoolMatch = true;
    }
  }
  EXPECT_EQ(avgPoolMatch, true);
}

TEST_F(matrix_diag_part_fusion_pass_test, matrix_diag_part_fusion_pass_test_4) {
  ge::Graph graph("matrix_diag_part_fusion_pass_test_4");
  auto matrix_diag_input_data = op::Data("matrix_diag_input_data");
  std::vector<int64_t> dims{128};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_INT8);
  matrix_diag_input_data.update_input_desc_x(tensorDesc);
  matrix_diag_input_data.update_output_desc_y(tensorDesc);
  auto matrix_diag_part_op = op::MatrixDiagPart("matrix_diag_0");
  matrix_diag_part_op.set_input_x(matrix_diag_input_data);
  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(matrix_diag_part_op);
  std::vector<Operator> inputs{matrix_diag_input_data};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatrixDiagPartFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool avgPoolMatch = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatrixDiagPartD") {
      avgPoolMatch = true;
    }
  }
  EXPECT_EQ(avgPoolMatch, true);
}

TEST_F(matrix_diag_part_fusion_pass_test, matrix_diag_part_fusion_pass_test_5) {
  ge::Graph graph("matrix_diag_part_fusion_pass_test_5");
  auto matrix_diag_input_data = op::Data("matrix_diag_input_data");
  std::vector<int64_t> dims{128};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_UINT8);
  matrix_diag_input_data.update_input_desc_x(tensorDesc);
  matrix_diag_input_data.update_output_desc_y(tensorDesc);
  auto matrix_diag_part_op = op::MatrixDiagPart("matrix_diag_0");
  matrix_diag_part_op.set_input_x(matrix_diag_input_data);
  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(matrix_diag_part_op);
  std::vector<Operator> inputs{matrix_diag_input_data};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatrixDiagPartFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool avgPoolMatch = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatrixDiagPartD") {
      avgPoolMatch = true;
    }
  }
  EXPECT_EQ(avgPoolMatch, true);
}

TEST_F(matrix_diag_part_fusion_pass_test, matrix_diag_part_fusion_pass_test_6) {
  ge::Graph graph("matrix_diag_part_fusion_pass_test_6");
  auto matrix_diag_input_data = op::Data("matrix_diag_input_data");
  std::vector<int64_t> dims{128};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_BOOL);
  matrix_diag_input_data.update_input_desc_x(tensorDesc);
  matrix_diag_input_data.update_output_desc_y(tensorDesc);
  auto matrix_diag_part_op = op::MatrixDiagPart("matrix_diag_0");
  matrix_diag_part_op.set_input_x(matrix_diag_input_data);
  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(matrix_diag_part_op);
  std::vector<Operator> inputs{matrix_diag_input_data};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatrixDiagPartFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool avgPoolMatch = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatrixDiagPartD") {
      avgPoolMatch = true;
    }
  }
  EXPECT_EQ(avgPoolMatch, false);
}