#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include "transformation_ops.h"
#include "nonlinear_fuc_ops.h"

using namespace ge;
using namespace op;

class affine_grid_fusion_test : public testing::Test {
  protected:
  static void SetUpTestCase() { std::cout << "affine_grid_fusion_test SetUp" << std::endl; }
  static void TearDownTestCase() { std::cout << "affine_grid_fusion_test TearDown" << std::endl; }
};

TEST_F(affine_grid_fusion_test, input_nd) {
  ge::Graph graph("input_nchw_graph");

  auto input_theta = op::Data("theta");
  std::vector<int64_t> dims_theta{2, 2, 3};
  ge::TensorDesc tensor_desc_x(ge::Shape(dims_theta), FORMAT_ND, DT_FLOAT);
  input_theta.update_input_desc_x(tensor_desc_x);
  input_theta.update_output_desc_y(tensor_desc_x);

  TensorDesc tensor_desc_output_size(ge::Shape({4}), FORMAT_ND, DT_INT32);
  Tensor output_size_tensor(tensor_desc_output_size);
  uint32_t *output_size_tensor_value = new uint32_t[4]{2, 3, 4, 5};
  output_size_tensor.SetData((uint8_t *) output_size_tensor_value, 4 * sizeof(uint32_t));
  
  auto input_output_size = op::Const("output_size").set_attr_value(output_size_tensor);

  auto affine_grid_op = op::AffineGrid("AffineGrid");
  affine_grid_op.set_input_theta(input_theta).set_input_output_size(input_output_size);

  auto relu_op = op::Relu("relu_op");
  relu_op.set_input_x(affine_grid_op);

  std::vector<Operator> inputs{input_theta};
  std::vector<Operator> outputs{relu_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("AffineGridFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool find_op = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchMatMul") {
      find_op = true;
    }
  }
  EXPECT_EQ(find_op, true);
  delete[] output_size_tensor_value;
}

TEST_F(affine_grid_fusion_test, input_int64) {
  ge::Graph graph("input_nchw_graph");

  auto input_theta = op::Data("theta");
  std::vector<int64_t> dims_theta{3, 2, 3};
  ge::TensorDesc tensor_desc_x(ge::Shape(dims_theta), FORMAT_ND, DT_INT64);
  input_theta.update_input_desc_x(tensor_desc_x);
  input_theta.update_output_desc_y(tensor_desc_x);

  TensorDesc tensor_desc_output_size(ge::Shape({4}), FORMAT_ND, DT_INT32);
  Tensor output_size_tensor(tensor_desc_output_size);
  uint32_t *output_size_tensor_value = new uint32_t[4]{3, 5, 7, 9};
  output_size_tensor.SetData((uint8_t *) output_size_tensor_value, 4 * sizeof(uint32_t));

  auto input_output_size = op::Const("output_size").set_attr_value(output_size_tensor);

  auto affine_grid_op = op::AffineGrid("AffineGrid");
  affine_grid_op.set_input_theta(input_theta).set_input_output_size(input_output_size);

  auto relu_op = op::Relu("relu_op");
  relu_op.set_input_x(affine_grid_op);

  std::vector<Operator> inputs{input_theta};
  std::vector<Operator> outputs{relu_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("AffineGridFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool find_op = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "AffineGrid") {
      find_op = true;
    }
  }
  EXPECT_EQ(find_op, true);
  delete[] output_size_tensor_value;
}