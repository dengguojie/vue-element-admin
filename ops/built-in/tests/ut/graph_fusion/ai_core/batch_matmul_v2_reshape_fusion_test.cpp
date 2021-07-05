#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "matrix_calculation_ops.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "nonlinear_fuc_ops.h"

using namespace ge;
using namespace op;

class batch_matmul_v2_reshape_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "batch_matmul_v2_reshape_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "batch_matmul_v2_reshape_fusion_test TearDown" << std::endl;
    }
};

TEST_F(batch_matmul_v2_reshape_fusion_test, batch_matmul_v2_reshape_fusion_test_1) {
    ge::Graph graph("batch_matmul_v2_reshape_fusion_test_1");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{12};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND,  DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{12, 3};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND,  DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{bmOP};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReshapeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, true);
}


TEST_F(batch_matmul_v2_reshape_fusion_test, batch_matmul_v2_reshape_fusion_test_2) {
    ge::Graph graph("batch_matmul_v2_reshape_fusion_test_2");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{3, 12};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND,  DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{12};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND,  DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{bmOP};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReshapeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, true);
}

TEST_F(batch_matmul_v2_reshape_fusion_test, batch_matmul_v2_reshape_fusion_test_3) {
    ge::Graph graph("batch_matmul_v2_reshape_fusion_test_3");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{3, 12};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND,  DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{12};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND,  DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    
    auto relu_op = op::Relu("Relu");
    relu_op.set_input_x(bmOP);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{relu_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReshapeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, true);
}