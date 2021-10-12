#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nonlinear_fuc_ops.h"
#include "array_ops.h"
#include "nn_pooling_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class adaptive_max_pool2d_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "adaptive_max_pool2d_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "adaptive_max_pool2d_fusion_test TearDown" << std::endl;
    }
};

TEST_F(adaptive_max_pool2d_fusion_test, adaptive_max_pool2d_fusion_test_1) {
    ge::Graph graph("adaptive_max_pool2d_fusion_test_1");

    auto input_x_data = op::Data("input_x_data");
    std::vector<int64_t> dims_x{1, 16, 3, 3};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
    input_x_data.update_input_desc_x(tensorDescX);
    input_x_data.update_output_desc_y(tensorDescX);

    auto adaptive_max_pool2d_op = op::AdaptiveMaxPool2d("adaptive_max_pool2d_0");
    adaptive_max_pool2d_op.set_input_x(input_x_data)
                  .set_attr_output_size({2, 2});

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(adaptive_max_pool2d_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{relu_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AdaptiveMaxPool2dFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findAdaptiveMaxPool2d = false;

    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
            findAdaptiveMaxPool2d = true;
        }
    }
    EXPECT_EQ(findAdaptiveMaxPool2d, false);
}

TEST_F(adaptive_max_pool2d_fusion_test, adaptive_max_pool2d_fusion_test_2) {
    ge::Graph graph("adaptive_max_pool2d_fusion_test_2");

    auto input_x_data = op::Data("input_x_data");
    std::vector<int64_t> dims_x{1, 16, 4, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
    input_x_data.update_input_desc_x(tensorDescX);
    input_x_data.update_output_desc_y(tensorDescX);

    auto adaptive_max_pool2d_op = op::AdaptiveMaxPool2d("adaptive_max_pool2d_0");
    adaptive_max_pool2d_op.set_input_x(input_x_data)
                  .set_attr_output_size({2, 2});

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(adaptive_max_pool2d_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{relu_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AdaptiveMaxPool2dFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findAdaptiveMaxPool2d = false;

    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
            findAdaptiveMaxPool2d = true;
        }
    }
    EXPECT_EQ(findAdaptiveMaxPool2d, true);
}

TEST_F(adaptive_max_pool2d_fusion_test, adaptive_max_pool2d_fusion_test_3) {
    ge::Graph graph("adaptive_max_pool2d_fusion_test_3");

    auto input_x_data = op::Data("input_x_data");
    std::vector<int64_t> dims_x{1, 6, 6, 16};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT);
    input_x_data.update_input_desc_x(tensorDescX);
    input_x_data.update_output_desc_y(tensorDescX);

    auto adaptive_max_pool2d_op = op::AdaptiveMaxPool2d("adaptive_max_pool2d_0");
    adaptive_max_pool2d_op.set_input_x(input_x_data)
                  .set_attr_output_size({2, 2});

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(adaptive_max_pool2d_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{relu_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AdaptiveMaxPool2dFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findAdaptiveMaxPool2d = false;

    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
            findAdaptiveMaxPool2d = true;
        }
    }
    EXPECT_EQ(findAdaptiveMaxPool2d, true);
}

