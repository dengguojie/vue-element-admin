#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "reduce_ops.h"

using namespace ge;
using namespace op;

class softmax_grad_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "softmax_grad_fusion SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "softmax_grad_fusion TearDown" << std::endl;
    }
};

TEST_F(softmax_grad_fusion_test, softmax_grad_fusion_test_1) {
    ge::Graph graph("softmax_grad_fusion_test_1");

    auto mul0 = op::Mul("mul0");
    std::vector<int64_t> dims_mul{2, 128, 128, 128, 4};
    ge::Shape shape_mul(dims_mul);
    ge::TensorDesc tensorDescData(shape_mul, ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    mul0.update_input_desc_x1(tensorDescData);
    mul0.update_input_desc_x2(tensorDescData);
    mul0.update_output_desc_y(tensorDescData);

    auto strideslicegrad = op::Data("strideslicegrad");
    auto transdata = op::Data("transdata");
    auto biasaddgrad = op::Data("biasaddgrad");
    strideslicegrad.update_input_desc_x(tensorDescData);
    strideslicegrad.update_output_desc_y(tensorDescData);
    transdata.update_input_desc_x(tensorDescData);
    transdata.update_output_desc_y(tensorDescData);
    biasaddgrad.update_input_desc_x(tensorDescData);
    biasaddgrad.update_output_desc_y(tensorDescData);

    std::vector<int64_t> dims_reducesumd{2, 128, 128, 128, 1};
    ge::Shape shape_reducesumd(dims_reducesumd);
    ge::TensorDesc tensor1DescData(shape_reducesumd, ge::FORMAT_NDHWC, ge::DT_FLOAT16);

    auto reducesumd = op::ReduceSumD("reducesumd");
    reducesumd.update_input_desc_x(tensorDescData);
    reducesumd.update_output_desc_y(tensor1DescData);

    auto sub = op::Sub("sub");
    sub.update_input_desc_x1(tensorDescData);
    sub.update_input_desc_x2(tensor1DescData);
    sub.update_output_desc_y(tensorDescData);

    auto mul1 = op::Mul("mul1");
    mul1.update_input_desc_x1(tensorDescData);
    mul1.update_input_desc_x2(tensorDescData);
    mul1.update_output_desc_y(tensorDescData);
    
    mul0.set_input_x1(transdata)
        .set_input_x2(strideslicegrad);
    reducesumd.set_input_x(mul0)
              .set_attr_axes({-1})
	      .set_attr_keep_dims(true);
    sub.set_input_x1(strideslicegrad)
       .set_input_x2(reducesumd);
    mul1.set_input_x1(transdata)
        .set_input_x2(sub);
    biasaddgrad.set_input_x(mul1);
    
    std::vector<Operator> inputs{strideslicegrad, transdata};
    std::vector<Operator> outputs{biasaddgrad};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("SoftmaxGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findsoftmaxgrad = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SoftmaxGrad") {
        findsoftmaxgrad = true;
        auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
        std::vector<int64_t> dims = output_desc.GetShape().GetDims();
        // EXPECT_EQ(output_range, expected_range);
        EXPECT_EQ(dims, dims_mul);
        }
    }
    EXPECT_EQ(findsoftmaxgrad, true);
}
