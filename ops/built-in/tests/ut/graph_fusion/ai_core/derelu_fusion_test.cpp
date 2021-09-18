#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nonlinear_fuc_ops.h"
#include "pad_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class derelu_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "inplace_add TearDown" << std::endl;
    }
};

TEST_F(derelu_fusion_test, derelu_fusion_test_1) {
    ge::Graph graph("derelu_fusion_test_1");
    std::vector<int64_t> dims{3, 32};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape);

    auto relu_input_data = op::Data("relu_input_data");
    relu_input_data.update_input_desc_x(tensorDesc);
    relu_input_data.update_output_desc_y(tensorDesc);

    auto relu_op = op::Relu("relu");
    relu_op.set_input_x(relu_input_data);

    std::vector<int64_t> grad_dims{-1, 32};
    ge::Shape grad_shape(grad_dims);
    ge::TensorDesc tensorDesc2(grad_shape);
    auto relu_grad_data = op::Data("relu_grad_data");
    relu_grad_data.update_input_desc_x(tensorDesc2);
    relu_grad_data.update_output_desc_y(tensorDesc2);

    auto reluGrad_op = op::ReluGrad("relugrad");
    reluGrad_op.set_input_gradients(relu_grad_data)
               .set_input_features(relu_op);

    std::vector<Operator> inputs{relu_input_data, relu_grad_data};
    std::vector<Operator> outputs{reluGrad_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("DreluFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

}