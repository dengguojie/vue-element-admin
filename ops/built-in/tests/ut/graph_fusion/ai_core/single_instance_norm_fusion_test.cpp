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

class single_instance_norm_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "single_instance_norm_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "single_instance_norm_fusion_test TearDown" << std::endl;
  }
};

TEST_F(single_instance_norm_fusion_test, single_instance_norm_fusion_test_1) {
  ge::Graph graph("single_instance_norm_fusion_test_1");

  auto in_input_x_data = op::Data("in_input_x_data");
  std::vector<int64_t> dims_x{1, 2, 3, 4, 4};
  ge::Shape shape_x(dims_x);
  ge::TensorDesc tensorDescX(shape_x, FORMAT_NCDHW, DT_FLOAT);
  in_input_x_data.update_input_desc_x(tensorDescX);
  in_input_x_data.update_output_desc_y(tensorDescX);

  auto in_input_gamma_data = op::Data("in_input_gamma_data");
  auto in_input_beta_data = op::Data("in_input_beta_data");
  std::vector<int64_t> dims_gamma{4};
  ge::Shape shape_gamma(dims_gamma);
  ge::TensorDesc tensorDescGamma(shape_gamma, FORMAT_ND, DT_FLOAT);
  in_input_gamma_data.update_input_desc_x(tensorDescGamma);
  in_input_gamma_data.update_output_desc_y(tensorDescGamma);

  in_input_beta_data.update_input_desc_x(tensorDescGamma);
  in_input_beta_data.update_output_desc_y(tensorDescGamma);

  auto in_op = op::InstanceNorm("instancenorm_0");
  in_op.set_input_x(in_input_x_data).set_input_gamma(in_input_gamma_data).set_input_beta(in_input_beta_data);

  auto relu_op = op::Relu("relu_op");
  relu_op.set_input_x(in_op, "y");

  std::vector<Operator> inputs{in_input_x_data, in_input_gamma_data, in_input_beta_data};
  std::vector<Operator> outputs{relu_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SingleInstanceNormFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findInreduce = false;
  bool findInupdate = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "INTrainingReduceV2") {
      findInreduce = true;
    }
    if (node->GetType() == "INTrainingUpdateV2") {
      findInupdate = true;
    }
  }
  EXPECT_EQ(findInreduce, true);
  EXPECT_EQ(findInupdate, true);
}

TEST_F(single_instance_norm_fusion_test, single_instance_norm_fusion_test_2) {
  ge::Graph graph("single_instance_norm_fusion_test_1");

  auto in_input_x_data = op::Data("in_input_x_data");
  std::vector<int64_t> dims_x{1, 2, 4, 4};
  ge::Shape shape_x(dims_x);
  ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT);
  in_input_x_data.update_input_desc_x(tensorDescX);
  in_input_x_data.update_output_desc_y(tensorDescX);

  auto in_input_gamma_data = op::Data("in_input_gamma_data");
  auto in_input_beta_data = op::Data("in_input_beta_data");
  std::vector<int64_t> dims_gamma{4};
  ge::Shape shape_gamma(dims_gamma);
  ge::TensorDesc tensorDescGamma(shape_gamma, FORMAT_ND, DT_FLOAT);
  in_input_gamma_data.update_input_desc_x(tensorDescGamma);
  in_input_gamma_data.update_output_desc_y(tensorDescGamma);

  in_input_beta_data.update_input_desc_x(tensorDescGamma);
  in_input_beta_data.update_output_desc_y(tensorDescGamma);

  auto in_op = op::InstanceNorm("instancenorm_0");
  in_op.set_input_x(in_input_x_data).set_input_gamma(in_input_gamma_data).set_input_beta(in_input_beta_data);

  auto relu_op = op::Relu("relu_op");
  relu_op.set_input_x(in_op, "y");

  std::vector<Operator> inputs{in_input_x_data, in_input_gamma_data, in_input_beta_data};
  std::vector<Operator> outputs{relu_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SingleInstanceNormFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findInreduce = false;
  bool findInupdate = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "INTrainingReduceV2") {
      findInreduce = true;
    }
    if (node->GetType() == "INTrainingUpdateV2") {
      findInupdate = true;
    }
  }
  EXPECT_EQ(findInreduce, true);
  EXPECT_EQ(findInupdate, true);
}
