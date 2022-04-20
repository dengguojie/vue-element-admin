#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class cumsum_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "cumsum_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "cumsum_fusion_test TearDown" << std::endl;
  }
};

TEST_F(cumsum_fusion_test, cumsum_fusion_test_1) {
    ge::Graph graph("cumsum_fusion_test_1");

    auto in_input_x_data = op::Data("diag_input_data");
    std::vector<int64_t> dims{4, 8, 900, 3};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto cum_op = op::CumsumD("cumsumd_0");
    cum_op.set_input_x(in_input_x_data)
        .set_attr_axis({-1});

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(cum_op);

    std::vector<Operator> inputs{in_input_x_data};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("CumFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findCum = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "CumsumD") {
            findCum = true;
        }
    }
    EXPECT_EQ(findCum, true);
}

TEST_F(cumsum_fusion_test, cumsum_fusion_test_2) {
    ge::Graph graph("cumsum_fusion_test_2");

    auto in_input_x_data = op::Data("diag_input_data");
    std::vector<int64_t> dims{1, 4, 8, 900, 3};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto cum_op = op::CumprodD("cumprod_0");
    cum_op.set_input_x(in_input_x_data)
        .set_attr_axis({-1});

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(cum_op);

    std::vector<Operator> inputs{in_input_x_data};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("CumFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findCum = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "CumprodD") {
            findCum = true;
        }
    }
    EXPECT_EQ(findCum, true);
}

TEST_F(cumsum_fusion_test, cumsum_fusion_test_3) {
    ge::Graph graph("cumsum_fusion_test_3");

    auto in_input_x_data = op::Data("diag_input_data");
    std::vector<int64_t> dims{1, 8732, 21};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto cum_op = op::CumsumD("cumsumd_0");
    cum_op.set_input_x(in_input_x_data)
        .set_attr_axis({-1});

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(cum_op);

    std::vector<Operator> inputs{in_input_x_data};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("CumFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findCum = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "CumsumD") {
            findCum = true;
        }
    }
    EXPECT_EQ(findCum, true);
}

TEST_F(cumsum_fusion_test, cumsum_fusion_test_4) {
    ge::Graph graph("cumsum_fusion_test_4");

    auto in_input_x_data = op::Data("diag_input_data");
    std::vector<int64_t> dims{8, 8732, 81};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto cum_op = op::CumprodD("cumprod_0");
    cum_op.set_input_x(in_input_x_data)
        .set_attr_axis({-1});

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(cum_op);

    std::vector<Operator> inputs{in_input_x_data};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("CumFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findCum = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "CumprodD") {
            findCum = true;
        }
    }
    EXPECT_EQ(findCum, true);
}
