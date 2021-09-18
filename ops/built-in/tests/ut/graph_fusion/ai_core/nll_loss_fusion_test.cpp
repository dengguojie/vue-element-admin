#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "math_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class nll_loss_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "nll_loss_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "nll_loss_fusion_test TearDown" << std::endl;
    }
};

TEST_F(nll_loss_fusion_test, nll_loss_fusion_test_1) {

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{1028, 1028};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // target
  auto data_target = op::Data("target");
  std::vector<int64_t> data_target_vec{1028};
  ge::Shape data_target_shape(data_target_vec);
  ge::TensorDesc data_target_desc(data_target_shape, FORMAT_ND, DT_INT32);
  data_target.update_input_desc_x(data_target_desc);
  data_target.update_output_desc_y(data_target_desc);

  // weight
  auto data_weight = op::Data("weight");
  std::vector<int64_t> data_weight_vec{1028};
  ge::Shape data_weight_shape(data_weight_vec);
  ge::TensorDesc data_weight_desc(data_weight_shape, FORMAT_ND, DT_FLOAT);
  data_weight.update_input_desc_x(data_weight_desc);
  data_weight.update_output_desc_y(data_weight_desc);

  // reduction
  string reduction = "sum";

  auto nll_loss_op = op::NLLLoss("NLLLoss");
  nll_loss_op.set_input_x(data_x)
             .set_input_target(data_target)
             .set_input_weight(data_weight)
             .set_attr_reduction(reduction);

  std::vector<Operator> inputs{data_x, data_target, data_weight};
  std::vector<Operator> outputs{nll_loss_op};

  ge::Graph graph("nll_loss_fusion_test_1");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("NLLLossFusionPass",
                                               fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool find_nll_loss = false;
  bool find_div = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "NLLLoss") {
            find_nll_loss = true;
        }

        if (node->GetType() == "Div") {
            find_div = true;
        }
    }

  EXPECT_EQ(find_nll_loss, true);
  EXPECT_EQ(find_div, false);
}

TEST_F(nll_loss_fusion_test, nll_loss_fusion_test_2) {

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{-1, -1};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // target
  auto data_target = op::Data("target");
  std::vector<int64_t> data_target_vec{-1};
  ge::Shape data_target_shape(data_target_vec);
  ge::TensorDesc data_target_desc(data_target_shape, FORMAT_ND, DT_INT32);
  data_target.update_input_desc_x(data_target_desc);
  data_target.update_output_desc_y(data_target_desc);

  // weight
  auto data_weight = op::Data("weight");
  std::vector<int64_t> data_weight_vec{-1};
  ge::Shape data_weight_shape(data_weight_vec);
  ge::TensorDesc data_weight_desc(data_weight_shape, FORMAT_ND, DT_FLOAT);
  data_weight.update_input_desc_x(data_weight_desc);
  data_weight.update_output_desc_y(data_weight_desc);

  // reduction
  string reduction = "sum";

  auto nll_loss_op = op::NLLLoss("NLLLoss");
  nll_loss_op.set_input_x(data_x)
             .set_input_target(data_target)
             .set_input_weight(data_weight)
             .set_attr_reduction(reduction);

  std::vector<Operator> inputs{data_x, data_target, data_weight};
  std::vector<Operator> outputs{nll_loss_op};

  ge::Graph graph("nll_loss_fusion_test_2");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("NLLLossFusionPass",
                                               fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool find_nll_loss = false;
  bool find_div = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "NLLLoss") {
            find_nll_loss = true;
        }

        if (node->GetType() == "Div") {
            find_div = true;
        }
    }

  EXPECT_EQ(find_nll_loss, true);
  EXPECT_EQ(find_div, false);
}

TEST_F(nll_loss_fusion_test, nll_loss_fusion_test_3) {

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{-1, -1};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // target
  auto data_target = op::Data("target");
  std::vector<int64_t> data_target_vec{-1};
  ge::Shape data_target_shape(data_target_vec);
  ge::TensorDesc data_target_desc(data_target_shape, FORMAT_ND, DT_INT32);
  data_target.update_input_desc_x(data_target_desc);
  data_target.update_output_desc_y(data_target_desc);

  // weight
  auto data_weight = op::Data("weight");
  std::vector<int64_t> data_weight_vec{-1};
  ge::Shape data_weight_shape(data_weight_vec);
  ge::TensorDesc data_weight_desc(data_weight_shape, FORMAT_ND, DT_FLOAT);
  data_weight.update_input_desc_x(data_weight_desc);
  data_weight.update_output_desc_y(data_weight_desc);

  // reduction
  string reduction = "mean";

  auto nll_loss_op = op::NLLLoss("NLLLoss");
  nll_loss_op.set_input_x(data_x)
             .set_input_target(data_target)
             .set_input_weight(data_weight)
             .set_attr_reduction(reduction);

  std::vector<Operator> inputs{data_x, data_target, data_weight};
  std::vector<Operator> outputs{nll_loss_op};

  ge::Graph graph("nll_loss_fusion_test_3");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("NLLLossFusionPass",
                                               fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool find_nll_loss = false;
  bool find_div = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "NLLLoss") {
            find_nll_loss = true;
        }

        if (node->GetType() == "Div") {
            find_div = true;
        }
    }

  EXPECT_EQ(find_nll_loss, true);
  EXPECT_EQ(find_div, true);
}

TEST_F(nll_loss_fusion_test, nll_loss_fusion_test_4) {

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{4210704, 21};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // target
  auto data_target = op::Data("target");
  std::vector<int64_t> data_target_vec{4210704};
  ge::Shape data_target_shape(data_target_vec);
  ge::TensorDesc data_target_desc(data_target_shape, FORMAT_ND, DT_INT32);
  data_target.update_input_desc_x(data_target_desc);
  data_target.update_output_desc_y(data_target_desc);

  // weight
  auto data_weight = op::Data("weight");
  std::vector<int64_t> data_weight_vec{21};
  ge::Shape data_weight_shape(data_weight_vec);
  ge::TensorDesc data_weight_desc(data_weight_shape, FORMAT_ND, DT_FLOAT);
  data_weight.update_input_desc_x(data_weight_desc);
  data_weight.update_output_desc_y(data_weight_desc);

  // reduction
  string reduction = "mean";

  auto nll_loss_op = op::NLLLoss("NLLLoss");
  nll_loss_op.set_input_x(data_x)
             .set_input_target(data_target)
             .set_input_weight(data_weight)
             .set_attr_reduction(reduction);

  std::vector<Operator> inputs{data_x, data_target, data_weight};
  std::vector<Operator> outputs{nll_loss_op};

  ge::Graph graph("nll_loss_fusion_test_4");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("NLLLossFusionPass",
                                               fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool find_nll_loss = false;
  bool find_div = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "NLLLoss") {
            find_nll_loss = true;
        }

        if (node->GetType() == "Div") {
            find_div = true;
        }
    }

  EXPECT_EQ(find_nll_loss, true);
  EXPECT_EQ(find_div, true);
}

TEST_F(nll_loss_fusion_test, nll_loss_fusion_test_5) {

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{4210704, 3};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // target
  auto data_target = op::Data("target");
  std::vector<int64_t> data_target_vec{4210704};
  ge::Shape data_target_shape(data_target_vec);
  ge::TensorDesc data_target_desc(data_target_shape, FORMAT_ND, DT_INT32);
  data_target.update_input_desc_x(data_target_desc);
  data_target.update_output_desc_y(data_target_desc);

  // weight
  auto data_weight = op::Data("weight");
  std::vector<int64_t> data_weight_vec{3};
  ge::Shape data_weight_shape(data_weight_vec);
  ge::TensorDesc data_weight_desc(data_weight_shape, FORMAT_ND, DT_FLOAT);
  data_weight.update_input_desc_x(data_weight_desc);
  data_weight.update_output_desc_y(data_weight_desc);

  // reduction
  string reduction = "mean";

  auto nll_loss_op = op::NLLLoss("NLLLoss");
  nll_loss_op.set_input_x(data_x)
             .set_input_target(data_target)
             .set_input_weight(data_weight)
             .set_attr_reduction(reduction);

  std::vector<Operator> inputs{data_x, data_target, data_weight};
  std::vector<Operator> outputs{nll_loss_op};

  ge::Graph graph("nll_loss_fusion_test_5");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("NLLLossFusionPass",
                                               fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool find_nll_loss = false;
  bool find_div = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "NLLLoss") {
            find_nll_loss = true;
        }

        if (node->GetType() == "Div") {
            find_div = true;
        }
    }

  EXPECT_EQ(find_nll_loss, true);
  EXPECT_EQ(find_div, false);
}