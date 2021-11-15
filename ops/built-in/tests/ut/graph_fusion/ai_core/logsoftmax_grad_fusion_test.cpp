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

class logsoftmax_grad_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "logsoftmax_grad_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "logsoftmax_grad_fusion_test TearDown" << std::endl;
    }
};

TEST_F(logsoftmax_grad_fusion_test, logsoftmax_grad_fusion_test_1) {
    ge::Graph graph("logsoftmax_grad_fusion_test_1");

    auto exp0 = op::Exp("exp0");
    std::vector<int64_t> dims_exp{2, 128, 128, 4};
    ge::Shape shape_exp(dims_exp);
    ge::TensorDesc tensorDescData(shape_exp, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    exp0.update_input_desc_x(tensorDescData);
    exp0.update_output_desc_y(tensorDescData);

    auto input0 = op::Data("input0");
    auto input1 = op::Data("input1");
    auto output0 = op::Data("output0");
    input0.update_input_desc_x(tensorDescData);
    input0.update_output_desc_y(tensorDescData);
    input1.update_input_desc_x(tensorDescData);
    input1.update_output_desc_y(tensorDescData);
    output0.update_input_desc_x(tensorDescData);
    output0.update_output_desc_y(tensorDescData);

    std::vector<int64_t> dims_reducesumd{2, 128, 128, 1};
    ge::Shape shape_reducesumd(dims_reducesumd);
    ge::TensorDesc tensor1DescData(shape_reducesumd, ge::FORMAT_NHWC, ge::DT_FLOAT16);

    auto reducesumd = op::ReduceSumD("reducesumd");
    reducesumd.update_input_desc_x(tensorDescData);
    reducesumd.update_output_desc_y(tensor1DescData);

    auto mul0 = op::Mul("mul1");
    mul0.update_input_desc_x1(tensorDescData);
    mul0.update_input_desc_x2(tensorDescData);
    mul0.update_output_desc_y(tensorDescData);

    auto sub = op::Sub("sub");
    sub.update_input_desc_x1(tensorDescData);
    sub.update_input_desc_x2(tensor1DescData);
    sub.update_output_desc_y(tensorDescData);

    exp0.set_input_x(input0);
    reducesumd.set_input_x(input1)
              .set_attr_axes({-1})
	          .set_attr_keep_dims(true);
    mul0.set_input_x1(exp0)
        .set_input_x2(reducesumd);
    sub.set_input_x1(input1)
       .set_input_x2(mul0);
    output0.set_input_x(sub);
    
    std::vector<Operator> inputs{input0, input1};
    std::vector<Operator> outputs{output0};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LogSoftmaxGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findlogsoftmaxgrad = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "LogSoftmaxGrad") {
        findlogsoftmaxgrad = true;
        }
    }
    EXPECT_EQ(findlogsoftmaxgrad, true);
}
