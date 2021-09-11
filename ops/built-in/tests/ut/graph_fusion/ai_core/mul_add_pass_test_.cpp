#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
using namespace op;

class mul_add_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "mul_add_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "mul_add_fusion_test TearDown" << std::endl;
    }
};

TEST_F(mul_add_fusion_test, mul_add_fusion_test_1) {
    ge::Graph graph("mul_add_fusion_test_1");
    auto input_data_1 = op::Data("input_data_1");
    std::vector<int64_t> dims{2, 128};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
    input_data_1.update_input_desc_x(tensorDesc);
    input_data_1.update_output_desc_y(tensorDesc);
    auto mul_op = op::Mul("bert/mul")
                      .set_input_x1(input_data_1)
                      .set_input_x2(input_data_1);
    auto add_op = op::Add("bert/add")
                      .set_input_x1(mul_op)
                      .set_input_x2(input_data_1);
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(add_op);
    std::vector<Operator> inputs{input_data_1};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("MulAddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool passMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "FusedMulAdd") {
            passMatch = true;
        }
    }
    EXPECT_EQ(passMatch, true);
}

TEST_F(mul_add_fusion_test, mul_add_fusion_test_2) {
    ge::Graph graph("mul_add_fusion_test_2");
    auto input_data_1 = op::Data("input_data_1");
    std::vector<int64_t> dims{2, 128};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
    input_data_1.update_input_desc_x(tensorDesc);
    input_data_1.update_output_desc_y(tensorDesc);
    auto mul_op = op::Mul("mul")
                      .set_input_x1(input_data_1)
                      .set_input_x2(input_data_1);
    auto add_op = op::Add("add")
                      .set_input_x1(mul_op)
                      .set_input_x2(input_data_1);
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(add_op);
    std::vector<Operator> inputs{input_data_1};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("MulAddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool passMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "FusedMulAdd") {
            passMatch = true;
        }
    }
    EXPECT_EQ(passMatch, false);
}
