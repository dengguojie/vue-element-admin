#include "gtest/gtest.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "nonlinear_fuc_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class softplus_tanh_mul_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "softplus_tanh_mul_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "softplus_tanh_mul_fusion_test TearDown" << std::endl;
    }
};

TEST_F(softplus_tanh_mul_fusion_test, softplus_tanh_mul_fusion_test1) {
  /**
   *    input -----
   *      |       |
   *   softplus   |
   *      |       |
   *    tanh      |
   *     |        |
   *    mul <-----
   *     |
   *   output
   *
   *
   */
  ge::Graph graph("softplus_tanh_mul_fusion_test1");

  auto input_op = op::Data("input");
  std::vector<int64_t> dims{900, 3};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
  input_op.update_input_desc_x(tensorDesc);
  input_op.update_output_desc_y(tensorDesc);

  auto softplus_op = op::Softplus("softplus");
  softplus_op.set_input_x(input_op);

  auto tanh_op = op::Tanh("tanh");
  tanh_op.set_input_x(softplus_op);

  auto mul_op = op::Mul("mul");
  mul_op.set_input_x1(tanh_op);
  mul_op.set_input_x2(input_op);

  auto output_op = op::Conv2D("output");
  output_op.set_input_x(mul_op);

  std::vector<Operator> inputs{input_op};
  std::vector<Operator> outputs{output_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SoftplusTanhMulPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findMish = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mish") {
      findMish = true;
    }
  }
  EXPECT_EQ(findMish, true);
}

TEST_F(softplus_tanh_mul_fusion_test, softplus_tanh_mul_fusion_test2) {
  /**
   *    input_1
   *      |
   *   softplus
   *      |
   *    tanh
   *     |
   *    mul <----- input_2
   *     |
   *   output
   *
   *
   */
  ge::Graph graph("softplus_tanh_mul_fusion_test2");

  std::vector<int64_t> dims{900, 3};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT);

  auto input_1_op = op::Data("input_1");
  input_1_op.update_input_desc_x(tensorDesc);
  input_1_op.update_output_desc_y(tensorDesc);

  auto input_2_op = op::Data("input_2");
  input_2_op.update_input_desc_x(tensorDesc);
  input_2_op.update_output_desc_y(tensorDesc);

  auto softplus_op = op::Softplus("softplus");
  softplus_op.set_input_x(input_1_op);

  auto tanh_op = op::Tanh("tanh");
  tanh_op.set_input_x(softplus_op);

  auto mul_op = op::Mul("mul");
  mul_op.set_input_x1(tanh_op);
  mul_op.set_input_x2(input_2_op);

  auto output_op = op::Conv2D("output");
  output_op.set_input_x(mul_op);

  std::vector<Operator> inputs{input_1_op, input_2_op};
  std::vector<Operator> outputs{output_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SoftplusTanhMulPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findMish = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mish") {
      findMish = true;
    }
  }
  EXPECT_EQ(findMish, false);
}

TEST_F(softplus_tanh_mul_fusion_test, softplus_tanh_mul_fusion_test3) {
  /**
   *           input -----
   *             |       |
   *          softplus   |
   *             |       |
   *           tanh      |
   *            |        |
   *           mul <-----
   *          /   \
   *    output_1  output_2
   *
   *
   */
  ge::Graph graph("softplus_tanh_mul_fusion_test3");

  auto input_op = op::Data("input");
  std::vector<int64_t> dims{900, 3};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
  input_op.update_input_desc_x(tensorDesc);
  input_op.update_output_desc_y(tensorDesc);

  auto softplus_op = op::Softplus("softplus");
  softplus_op.set_input_x(input_op);

  auto tanh_op = op::Tanh("tanh");
  tanh_op.set_input_x(softplus_op);

  auto mul_op = op::Mul("mul");
  mul_op.set_input_x1(tanh_op);
  mul_op.set_input_x2(input_op);

  auto output_1_op = op::Conv2D("output_1");
  output_1_op.set_input_x(mul_op);

  auto output_2_op = op::Conv2D("output_2");
  output_2_op.set_input_x(mul_op);

  std::vector<Operator> inputs{input_op};
  std::vector<Operator> outputs{output_1_op, output_2_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SoftplusTanhMulPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findMish = false;
  int outCounts = 0;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Mish") {
      findMish = true;
      outCounts = node->GetOutDataNodesSize();
    }
  }
  EXPECT_EQ(findMish, true);
  EXPECT_EQ(outCounts, 2);
}
