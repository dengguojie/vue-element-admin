#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "matrix_calculation_ops.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "nonlinear_fuc_ops.h"

using namespace ge;
using namespace op;

class batch_matmul_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "batch_matmul_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "batch_matmul_fusion_test TearDown" << std::endl;
  }
};

TEST_F(batch_matmul_fusion_test, batch_matmul_fusion_test_1) {
  // The first part: using IR for composition, pay attention to the attribute description of input and output
  ge::Graph graph("batch_matmul_fusion_test_1");
  auto batch_matmul_input_data1 = op::Data("batch_matmul_input_data1");
  auto batch_matmul_input_data2 = op::Data("batch_matmul_input_data2");

  std::vector<int64_t> dims{3, 4};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);

  std::vector<int64_t> dims1{4, 4};
  ge::Shape shape1(dims1);
  ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto batch_matmul_op = op::BatchMatMul("batch_matmul")
                             .set_input_x1(batch_matmul_input_data1)
                             .set_input_x2(batch_matmul_input_data2)
                             .set_attr_adj_x1(false)
                             .set_attr_adj_x2(false);

  batch_matmul_op.update_input_desc_x1(tensorDesc);
  batch_matmul_op.update_input_desc_x2(tensorDesc1);
  batch_matmul_op.update_output_desc_y(tensorDesc);

  std::vector<Operator> inputs{batch_matmul_input_data1, batch_matmul_input_data2};
  std::vector<Operator> outputs{batch_matmul_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatmulFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool matmul_match = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMul") {
      matmul_match = true;
    }
  }
  EXPECT_EQ(matmul_match, true);
}