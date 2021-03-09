#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include "nn_pooling_ops.h"
#include "nonlinear_fuc_ops.h"

using namespace ge;
using namespace op;

class adaptive_avgpool2d_grad_fusion_test : public testing::Test {
   protected:
    static void SetUpTestCase() { std::cout << "adaptive_avgpool2d_grad_fusion_test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "adaptive_avgpool2d_grad_fusion_test TearDown" << std::endl; }
};

TEST_F(adaptive_avgpool2d_grad_fusion_test, input_nd) {
    ge::Graph graph("input_nchw_graph");

    auto input_x_data = op::Data("input_x_data");
    std::vector<int64_t> dims_x{2, 3, 5, 6};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND, DT_FLOAT);
    input_x_data.update_input_desc_x(tensorDescX);
    input_x_data.update_output_desc_y(tensorDescX);

    auto adaptive_avgpool2d_grad_op = op::AdaptiveAvgPool2dGrad("adaptive_avgpool2d_grad");
    adaptive_avgpool2d_grad_op.set_input_input_grad(input_x_data).set_attr_orig_input_shape({2, 3, 7, 8});

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(adaptive_avgpool2d_grad_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{relu_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AdaptiveAvgPoolGradFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                                *compute_graph_ptr);

    bool findAdaptiveMaxPool2d = false;

    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            findAdaptiveMaxPool2d = true;
        }
    }
    EXPECT_EQ(findAdaptiveMaxPool2d, true);
}
