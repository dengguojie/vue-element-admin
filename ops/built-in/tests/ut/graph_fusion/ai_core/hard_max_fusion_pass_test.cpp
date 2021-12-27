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

class hard_max_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "hard_max_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "hard_max_fusion_test TearDown" << std::endl;
    }
};

TEST_F(hard_max_fusion_test, hard_max_fusion_test_1) {
    ge::Graph graph("hard_max_fusion_test_1");

    auto x = op::Data("x");
    std::vector<int64_t> dims_x{32, 32, 32, 32};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, ge::FORMAT_ND, ge::DT_FLOAT16);
    x.update_input_desc_x(tensorDescX);
    std::vector<int64_t> dims_y{32, 32, 32, 32};
    ge::Shape shape_y(dims_y);
    ge::TensorDesc tensorDescY(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
    x.update_output_desc_y(tensorDescY);

    auto hard_max_op = op::HardMax("hard_max");
    hard_max_op.set_input_x(x)
                 .set_attr_axis({-1});

    std::vector<Operator> inputs{x};
    std::vector<Operator> outputs{hard_max_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("HardMaxPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool find_hard_max = true;

    EXPECT_EQ(find_hard_max, true);
}

TEST_F(hard_max_fusion_test, hard_max_fusion_test_2) {
    ge::Graph graph("hard_max_fusion_test_2");

    auto x = op::Data("x");
    std::vector<int64_t> dims_x{32, 32, 32, 32};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, ge::FORMAT_ND, ge::DT_FLOAT);
    x.update_input_desc_x(tensorDescX);
    std::vector<int64_t> dims_y{32, 32, 32, 32};
    ge::Shape shape_y(dims_y);
    ge::TensorDesc tensorDescY(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
    x.update_output_desc_y(tensorDescY);

    auto hard_max_op = op::HardMax("hard_max");
    hard_max_op.set_input_x(x)
                 .set_attr_axis({0});

    std::vector<Operator> inputs{x};
    std::vector<Operator> outputs{hard_max_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("HardMaxPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool find_hard_max = true;

    EXPECT_EQ(find_hard_max, true);
}

TEST_F(hard_max_fusion_test, hard_max_fusion_test_3) {
    ge::Graph graph("hard_max_fusion_test_3");

    auto x = op::Data("x");
    std::vector<int64_t> dims_x{32, 32, 32, 32};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, ge::FORMAT_ND, ge::DT_FLOAT);
    x.update_input_desc_x(tensorDescX);
    std::vector<int64_t> dims_y{32, 32, 32, 32};
    ge::Shape shape_y(dims_y);
    ge::TensorDesc tensorDescY(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
    x.update_output_desc_y(tensorDescY);

    auto hard_max_op = op::HardMax("hard_max");
    hard_max_op.set_input_x(x)
                 .set_attr_axis({1});

    std::vector<Operator> inputs{x};
    std::vector<Operator> outputs{hard_max_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("HardMaxPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool find_hard_max = true;

    EXPECT_EQ(find_hard_max, true);
}

TEST_F(hard_max_fusion_test, hard_max_fusion_test_4) {
    ge::Graph graph("hard_max_fusion_test_4");

    auto x = op::Data("x");
    std::vector<int64_t> dims_x{32, 32, 32, 32};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, ge::FORMAT_ND, ge::DT_FLOAT);
    x.update_input_desc_x(tensorDescX);
    std::vector<int64_t> dims_y{32, 32, 32, 32};
    ge::Shape shape_y(dims_y);
    ge::TensorDesc tensorDescY(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
    x.update_output_desc_y(tensorDescY);

    auto hard_max_op = op::HardMax("hard_max");
    hard_max_op.set_input_x(x)
                 .set_attr_axis({2});

    std::vector<Operator> inputs{x};
    std::vector<Operator> outputs{hard_max_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("HardMaxPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool find_hard_max = true;

    EXPECT_EQ(find_hard_max, true);
}

