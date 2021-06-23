#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nonlinear_fuc_ops.h"
#include "state_ops.h"
#include "nn_norm_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
using namespace op;

class single_instance_norm_grad_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "single_instance_norm_grad_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "single_instance_norm_grad_fusion_test TearDown" << std::endl;
  }
};

TEST_F(single_instance_norm_grad_fusion_test, single_instance_norm_grad_fusion_test_1) {
  ge::Graph graph("single_instance_norm_grad_fusion_test_1");

  auto in_input_dy_data = op::Data("in_input_dy_data");
  auto in_input_x_data = op::Data("in_input_x_data");
  std::vector<int64_t> dims_dy{1, 2, 3, 4, 4};
  ge::Shape shape_dy(dims_dy);
  ge::TensorDesc tensorDescDy(shape_dy, FORMAT_NCDHW, DT_FLOAT);
  in_input_dy_data.update_input_desc_x(tensorDescDy);
  in_input_dy_data.update_output_desc_y(tensorDescDy);
  in_input_x_data.update_input_desc_x(tensorDescDy);
  in_input_x_data.update_output_desc_y(tensorDescDy);

  auto in_input_var_data = op::Data("in_input_var_data");
  auto in_input_mean_data = op::Data("in_input_mean_data");
  std::vector<int64_t> dims_var{1, 2, 1, 1, 1};
  ge::Shape shape_var(dims_var);
  ge::TensorDesc tensorDescVar(shape_var, FORMAT_NCDHW, DT_FLOAT);
  in_input_var_data.update_input_desc_x(tensorDescVar);
  in_input_var_data.update_output_desc_y(tensorDescVar);
  in_input_mean_data.update_input_desc_x(tensorDescVar);
  in_input_mean_data.update_output_desc_y(tensorDescVar);

  auto in_input_gamma_data = op::Data("in_input_gamma_data");
  std::vector<int64_t> dims_gamma{4};
  ge::Shape shape_gamma(dims_gamma);
  ge::TensorDesc tensorDescGamma(shape_gamma, FORMAT_ND, DT_FLOAT);
  in_input_gamma_data.update_input_desc_x(tensorDescGamma);
  in_input_gamma_data.update_output_desc_y(tensorDescGamma);

  auto in_op = op::InstanceNormGrad("instancenormgrad_0");
  in_op.set_input_dy(in_input_dy_data)
      .set_input_x(in_input_x_data)
      .set_input_variance(in_input_var_data)
      .set_input_mean(in_input_mean_data)
      .set_input_gamma(in_input_gamma_data);

  auto relu_op = op::Relu("relu_op");
  relu_op.set_input_x(in_op, "y");

  std::vector<Operator> inputs{in_input_dy_data, in_input_x_data, in_input_var_data, in_input_mean_data,
                               in_input_gamma_data};
  std::vector<Operator> outputs{relu_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SingleInstanceNormGradFusion", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findINTrainingUpdateGrad = false;
  bool findINTrainingReduceGrad = false;
  bool findINTrainingUpdateGradGammaBeta = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "INTrainingUpdateGrad") {
      findINTrainingUpdateGrad = true;
    }
    if (node->GetType() == "INTrainingReduceGrad") {
      findINTrainingReduceGrad = true;
    }
    if (node->GetType() == "INTrainingUpdateGradGammaBeta") {
      findINTrainingUpdateGradGammaBeta = true;
    }
  }
  EXPECT_EQ(findINTrainingUpdateGrad, true);
  EXPECT_EQ(findINTrainingReduceGrad, true);
  EXPECT_EQ(findINTrainingUpdateGradGammaBeta, true);
}

TEST_F(single_instance_norm_grad_fusion_test, single_instance_norm_grad_fusion_test_2) {
  ge::Graph graph("single_instance_norm_grad_fusion_test_2");

  auto in_input_dy_data = op::Data("in_input_dy_data");
  auto in_input_x_data = op::Data("in_input_x_data");
  std::vector<int64_t> dims_dy{1, 2, 4, 4};
  ge::Shape shape_dy(dims_dy);
  ge::TensorDesc tensorDescDy(shape_dy, FORMAT_NCDHW, DT_FLOAT);
  in_input_dy_data.update_input_desc_x(tensorDescDy);
  in_input_dy_data.update_output_desc_y(tensorDescDy);
  in_input_x_data.update_input_desc_x(tensorDescDy);
  in_input_x_data.update_output_desc_y(tensorDescDy);

  auto in_input_var_data = op::Data("in_input_var_data");
  auto in_input_mean_data = op::Data("in_input_mean_data");
  std::vector<int64_t> dims_var{1, 2, 1, 1};
  ge::Shape shape_var(dims_var);
  ge::TensorDesc tensorDescVar(shape_var, FORMAT_NCDHW, DT_FLOAT);
  in_input_var_data.update_input_desc_x(tensorDescVar);
  in_input_var_data.update_output_desc_y(tensorDescVar);
  in_input_mean_data.update_input_desc_x(tensorDescVar);
  in_input_mean_data.update_output_desc_y(tensorDescVar);

  auto in_input_gamma_data = op::Data("in_input_gamma_data");
  std::vector<int64_t> dims_gamma{4};
  ge::Shape shape_gamma(dims_gamma);
  ge::TensorDesc tensorDescGamma(shape_gamma, FORMAT_ND, DT_FLOAT);
  in_input_gamma_data.update_input_desc_x(tensorDescGamma);
  in_input_gamma_data.update_output_desc_y(tensorDescGamma);

  auto in_op = op::InstanceNormGrad("instancenormgrad_0");
  in_op.set_input_dy(in_input_dy_data)
      .set_input_x(in_input_x_data)
      .set_input_variance(in_input_var_data)
      .set_input_mean(in_input_mean_data)
      .set_input_gamma(in_input_gamma_data);

  auto relu_op = op::Relu("relu_op");
  relu_op.set_input_x(in_op, "y");

  std::vector<Operator> inputs{in_input_dy_data, in_input_x_data, in_input_var_data, in_input_mean_data,
                               in_input_gamma_data};
  std::vector<Operator> outputs{relu_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SingleInstanceNormGradFusion", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findINTrainingUpdateGrad = false;
  bool findINTrainingReduceGrad = false;
  bool findINTrainingUpdateGradGammaBeta = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "INTrainingUpdateGrad") {
      findINTrainingUpdateGrad = true;
    }
    if (node->GetType() == "INTrainingReduceGrad") {
      findINTrainingReduceGrad = true;
    }
    if (node->GetType() == "INTrainingUpdateGradGammaBeta") {
      findINTrainingUpdateGradGammaBeta = true;
    }
  }
  EXPECT_EQ(findINTrainingUpdateGrad, true);
  EXPECT_EQ(findINTrainingReduceGrad, true);
  EXPECT_EQ(findINTrainingUpdateGradGammaBeta, true);
}
