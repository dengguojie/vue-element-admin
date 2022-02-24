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

class matrix_set_diag_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "matrix_set_diag_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "matrix_set_diag_fusion_pass_test TearDown" << std::endl;
  }
};

TEST_F(matrix_set_diag_fusion_pass_test, matrix_set_diag_fusion_pass_test_1) {
  ge::Graph graph("matrix_set_diag_fusion_pass_test_1");
  auto matrix_set_diag_input_data = op::Data("matrix_diag_input_data");
  auto matrix_set_diag_input_data1 = op::Data("matrix_diag_input_data1");
  std::vector<int64_t> dims{128,32,16,32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
  matrix_set_diag_input_data.update_input_desc_x(tensorDesc);
  matrix_set_diag_input_data.update_output_desc_y(tensorDesc);
  std::vector<int64_t> dims1{128,32,16};
  ge::Shape shape1(dims1);
  ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_NHWC, ge::DT_FLOAT);
  matrix_set_diag_input_data1.update_input_desc_x(tensorDesc1);
  matrix_set_diag_input_data1.update_output_desc_y(tensorDesc1);

  auto matrix_set_diag_op = op::MatrixSetDiag("matrix_set_diag_part0");
  matrix_set_diag_op.set_input_x(matrix_set_diag_input_data);
  matrix_set_diag_op.set_input_diagonal(matrix_set_diag_input_data1);

  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(matrix_set_diag_op);

  std::vector<Operator> inputs{matrix_set_diag_input_data, matrix_set_diag_input_data1};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatrixSetDiagFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool OpMatch = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatrixSetDiagD") {
      OpMatch = true;
    }
  }
  EXPECT_EQ(OpMatch, true);
}
