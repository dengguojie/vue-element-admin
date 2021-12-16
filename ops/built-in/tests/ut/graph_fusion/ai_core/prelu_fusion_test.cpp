#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "nonlinear_fuc_ops.h"
#include "elewise_calculation_ops.h"

using namespace ge;
using namespace op;

class prelu_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "prelu_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "prelu_fusion_test TearDown" << std::endl;
    }
};
TEST_F(prelu_fusion_test, prelu_fusion_test_1) {
    ge::Graph graph("prelu_fusion_test_1");
    auto xdata = op::Data("x");
    std::vector<int64_t> dims{10, 10, 10, 10, 10};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc0(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    xdata.update_input_desc_x(tensorDesc0);
    xdata.update_output_desc_y(tensorDesc0);

    auto xdata1 = op::Data("x1");
    std::vector<int64_t> dims1{10, 10, 10, 10, 10};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT);
    xdata1.update_input_desc_x(tensorDesc1);
    xdata1.update_output_desc_y(tensorDesc1);

    auto neg_op1 = op::Neg("Neg_1");
    neg_op1.set_input_x(xdata);

    auto relu_op1 = op::Relu("Relu_1");
    relu_op1.set_input_x(neg_op1);

    auto neg_op0 = op::Neg("Neg_0");
    neg_op0.set_input_x(xdata1);

    auto mul_op = op::Mul("Mul_1");
    mul_op.set_input_x1(neg_op0)
        .set_input_x2(relu_op1);
    
    auto relu_op0 = op::Relu("Relu_0");
    relu_op0.set_input_x(xdata);
    
    auto add_op0 = op::Add("Add_0");
    add_op0.set_input_x1(mul_op)
        .set_input_x2(relu_op0);


    std::vector<Operator> inputs{xdata,xdata1};
    std::vector<Operator> outputs{add_op0};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PReluFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findPRelu = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "PRelu") {
            findPRelu = true;
        }
    }
    EXPECT_EQ(findPRelu, true);
}
