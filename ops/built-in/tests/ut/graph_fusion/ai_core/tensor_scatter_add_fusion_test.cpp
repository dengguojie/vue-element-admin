#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "matrix_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class tensor_scatter_add_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "tensor_scatter_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "tensor_scatter_add TearDown" << std::endl;
    }
};

TEST_F(tensor_scatter_add_fusion_test, tensor_scatter_add_fusion_test_1) {
    ge::Graph graph("tensor_scatter_add_fusion_test_1");

    auto tensor_scatter_add_input_x = op::Data("tensor_scatter_add_input_x");
    std::vector<int64_t> dims_1{33,5};
    ge::Shape shape1(dims_1);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    tensor_scatter_add_input_x.update_input_desc_x(tensorDesc1);
    tensor_scatter_add_input_x.update_output_desc_y(tensorDesc1);

    auto tensor_scatter_add_input_indices = op::Data("tensor_scatter_add_input_indices");
    std::vector<int64_t> dims_2{33,25,1};
    ge::Shape shape2(dims_2);
    ge::TensorDesc tensorDesc2(shape2, ge::FORMAT_NHWC, ge::DT_INT32);
    tensor_scatter_add_input_indices.update_input_desc_x(tensorDesc2);
    tensor_scatter_add_input_indices.update_output_desc_y(tensorDesc2);

    auto tensor_scatter_add_input_updates = op::Data("tensor_scatter_add_input_updates");
    std::vector<int64_t> dims_3{33,25,5};
    ge::Shape shape3(dims_3);
    ge::TensorDesc tensorDesc3(shape3, ge::FORMAT_NHWC, ge::DT_FLOAT);
    tensor_scatter_add_input_updates.update_input_desc_x(tensorDesc3);
    tensor_scatter_add_input_updates.update_output_desc_y(tensorDesc3);

    auto tensor_scatter_add_op = op::TensorScatterAdd("TensorScatterAdd_0");
    tensor_scatter_add_op.set_input_x(tensor_scatter_add_input_x);
    tensor_scatter_add_op.set_input_indices(tensor_scatter_add_input_indices);
    tensor_scatter_add_op.set_input_updates(tensor_scatter_add_input_updates);

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(tensor_scatter_add_op);
    std::vector<Operator> inputs{tensor_scatter_add_input_x, tensor_scatter_add_input_indices, tensor_scatter_add_input_updates};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TensorScatterAddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool TensorScatterAddMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TensorMove") {
            TensorScatterAddMatch = true;
        }
    }
    EXPECT_EQ(TensorScatterAddMatch, true);
}

TEST_F(tensor_scatter_add_fusion_test, tensor_scatter_add_fusion_test_2) {
    ge::Graph graph("tensor_scatter_add_fusion_test_2");

    auto tensor_scatter_add_input_x = op::Data("tensor_scatter_add_input_x");
    std::vector<int64_t> dims_1{13,5};
    ge::Shape shape1(dims_1);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    tensor_scatter_add_input_x.update_input_desc_x(tensorDesc1);
    tensor_scatter_add_input_x.update_output_desc_y(tensorDesc1);

    auto tensor_scatter_add_input_indices = op::Data("tensor_scatter_add_input_indices");
    std::vector<int64_t> dims_2{13,11,1};
    ge::Shape shape2(dims_2);
    ge::TensorDesc tensorDesc2(shape2, ge::FORMAT_NHWC, ge::DT_INT32);
    tensor_scatter_add_input_indices.update_input_desc_x(tensorDesc2);
    tensor_scatter_add_input_indices.update_output_desc_y(tensorDesc2);

    auto tensor_scatter_add_input_updates = op::Data("tensor_scatter_add_input_updates");
    std::vector<int64_t> dims_3{13,11,5};
    ge::Shape shape3(dims_3);
    ge::TensorDesc tensorDesc3(shape3, ge::FORMAT_NHWC, ge::DT_FLOAT);
    tensor_scatter_add_input_updates.update_input_desc_x(tensorDesc3);
    tensor_scatter_add_input_updates.update_output_desc_y(tensorDesc3);

    auto tensor_scatter_add_op = op::TensorScatterAdd("TensorScatterAdd_0");
    tensor_scatter_add_op.set_input_x(tensor_scatter_add_input_x);
    tensor_scatter_add_op.set_input_indices(tensor_scatter_add_input_indices);
    tensor_scatter_add_op.set_input_updates(tensor_scatter_add_input_updates);

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(tensor_scatter_add_op);
    std::vector<Operator> inputs{tensor_scatter_add_input_x, tensor_scatter_add_input_indices, tensor_scatter_add_input_updates};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TensorScatterAddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool TensorScatterAddMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TensorMove") {
            TensorScatterAddMatch = true;
        }
    }
    EXPECT_EQ(TensorScatterAddMatch, true);
}