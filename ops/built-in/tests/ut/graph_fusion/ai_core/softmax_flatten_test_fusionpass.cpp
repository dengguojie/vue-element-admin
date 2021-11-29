#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_norm_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class softmax_flatten_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "softmax_flatten_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "softmax_flatten_fusion_test TearDown" << std::endl;
  }
};

TEST_F(softmax_flatten_fusion_test, softmax_flatten_fusion_test_1) {
    ge::Graph graph("softmax_flatten_fusion_test_1");

    auto in_input_x_data = op::Data("diag_input_data");
    std::vector<int64_t> dims{1, 2, 208, 4};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto softmax_op = op::SoftmaxV2("softmaxv2_0");
    softmax_op.set_input_x(in_input_x_data).set_attr_axes({-1});
    softmax_op.SetAttr("need_fusion", 1);

    std::vector<Operator> inputs{in_input_x_data};
    std::vector<Operator> outputs{softmax_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ASoftmaxFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findSoftmax = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SoftmaxV2") {
            findSoftmax = true;
        }
    }
    EXPECT_EQ(findSoftmax, true);
}

TEST_F(softmax_flatten_fusion_test, softmax_flatten_fusion_test_2) {
    ge::Graph graph("softmax_flatten_fusion_test_1");

    auto in_input_x_data = op::Data("diag_input_data");
    std::vector<int64_t> dims{1, 2, 208, 4};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto softmax_op = op::SoftmaxV2("softmaxv2_0");
    softmax_op.set_input_x(in_input_x_data).set_attr_axes({2});
    softmax_op.SetAttr("need_fusion", 1);

    std::vector<Operator> inputs{in_input_x_data};
    std::vector<Operator> outputs{softmax_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ASoftmaxFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findSoftmax = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SoftmaxV2") {
            findSoftmax = true;
        }
    }
    EXPECT_EQ(findSoftmax, true);
}
