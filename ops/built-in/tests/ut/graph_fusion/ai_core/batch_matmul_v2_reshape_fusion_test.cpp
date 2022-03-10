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
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{12, 3};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
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
    fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReshapeFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

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
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{12};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
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
    fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReshapeFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

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
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{12};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
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
    fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReshapeFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, true);
}

TEST_F(batch_matmul_v2_reshape_fusion_test, batch_matmul_v2_reshape_fusion_test_4) {
    ge::Graph graph("batch_matmul_v2_reshape_fusion_test_4");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{3, 12};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{12};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto X3Data = op::Data("x3");
    std::vector<int64_t> dims_x3{3};
    ge::Shape shape_x3(dims_x3);
    ge::TensorDesc tensorDescX3(shape_x3, FORMAT_ND, DT_FLOAT16);
    X3Data.update_input_desc_x(tensorDescX3);
    X3Data.update_output_desc_y(tensorDescX3);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);

    auto add_op = op::Add("Add");
    add_op.set_input_x1(X3Data);
    add_op.set_input_x2(bmOP);

    std::vector<Operator> inputs{X1Data, X2Data, X3Data};
    std::vector<Operator> outputs{add_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReshapeFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, true);
}

TEST_F(batch_matmul_v2_reshape_fusion_test, batch_matmul_v2_reshape_fusion_test_5)
{
    ge::Graph graph("batch_matmul_v2_reshape_fusion_test_5");

    // BatchMatMulV2 --> Add --> Output
    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{128, 16, 1775};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{1775, 1981};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);

    auto X3Data = op::Data("x3");
    std::vector<int64_t> dims_x3{1981};
    ge::Shape shape_x3(dims_x3);
    ge::TensorDesc tensorDescX3(shape_x3, FORMAT_ND, DT_FLOAT16);
    X3Data.update_input_desc_x(tensorDescX3);
    X3Data.update_output_desc_y(tensorDescX3);

    auto add_op = op::Add("Add");
    add_op.set_input_x1(bmOP);
    add_op.set_input_x2(X3Data);

    // Additional test node
    auto X4Data = op::Data("x4");
    std::vector<int64_t> dims_x4{2048, 1981};
    ge::Shape shape_x4(dims_x4);
    ge::TensorDesc tensorDescX4(shape_x4, FORMAT_ND, DT_FLOAT16);
    X4Data.update_input_desc_x(tensorDescX4);
    X4Data.update_output_desc_y(tensorDescX4);

    auto mul_op = op::Mul("Mul");
    mul_op.set_input_x1(add_op);
    mul_op.set_input_x2(X4Data);

    std::vector<Operator> inputs{X1Data, X2Data, X3Data, X4Data};
    std::vector<Operator> outputs{mul_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReshapeFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"Reshape", 2}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

TEST_F(batch_matmul_v2_reshape_fusion_test, batch_matmul_v2_reshape_fusion_test_6)
{
    ge::Graph graph("batch_matmul_v2_reshape_fusion_test_6");

    // BatchMatMulV2 --> Add --> Add --> Output
    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{128, 16, 1775};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{1775, 1981};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);

    auto X3Data = op::Data("x3");
    std::vector<int64_t> dims_x3{1981};
    ge::Shape shape_x3(dims_x3);
    ge::TensorDesc tensorDescX3(shape_x3, FORMAT_ND, DT_FLOAT16);
    X3Data.update_input_desc_x(tensorDescX3);
    X3Data.update_output_desc_y(tensorDescX3);

    auto add_op = op::Add("Add");
    add_op.set_input_x1(bmOP);
    add_op.set_input_x2(X3Data);

    auto X4Data = op::Data("x4");
    std::vector<int64_t> dims_x4{128, 16, 1981};
    ge::Shape shape_x4(dims_x4);
    ge::TensorDesc tensorDescX4(shape_x4, FORMAT_ND, DT_FLOAT16);
    X4Data.update_input_desc_x(tensorDescX4);
    X4Data.update_output_desc_y(tensorDescX4);

    auto add_op_2 = op::Add("Add_2");
    add_op_2.set_input_x1(add_op);
    add_op_2.set_input_x2(X4Data);

    // Additional test node
    auto X5Data = op::Data("x5");
    std::vector<int64_t> dims_x5{2048, 1981};
    ge::Shape shape_x5(dims_x5);
    ge::TensorDesc tensorDescX5(shape_x5, FORMAT_ND, DT_FLOAT16);
    X5Data.update_input_desc_x(tensorDescX5);
    X5Data.update_output_desc_y(tensorDescX5);

    auto mul_op = op::Mul("Mul");
    mul_op.set_input_x1(add_op_2);
    mul_op.set_input_x2(X5Data);

    std::vector<Operator> inputs{X1Data, X2Data, X3Data, X4Data, X5Data};
    std::vector<Operator> outputs{mul_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReshapeFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"Reshape", 3}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

TEST_F(batch_matmul_v2_reshape_fusion_test, batch_matmul_v2_reshape_fusion_test_7)
{
    ge::Graph graph("batch_matmul_v2_reshape_fusion_test_7");

   /*
    * BatchMatMulV2 --> Add --> Mul --> Sigmoid --> Mul --> Output
    *                    \__________________________/
    */
    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{128, 16, 1775};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{1775, 1981};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);

    auto X3Data = op::Data("x3");
    std::vector<int64_t> dims_x3{1981};
    ge::Shape shape_x3(dims_x3);
    ge::TensorDesc tensorDescX3(shape_x3, FORMAT_ND, DT_FLOAT16);
    X3Data.update_input_desc_x(tensorDescX3);
    X3Data.update_output_desc_y(tensorDescX3);

    auto add_op = op::Add("Add");
    add_op.set_input_x1(bmOP);
    add_op.set_input_x2(X3Data);

    auto X4Data = op::Data("x4");
    std::vector<int64_t> dims_x4{128, 16, 1981};
    ge::Shape shape_x4(dims_x4);
    ge::TensorDesc tensorDescX4(shape_x4, FORMAT_ND, DT_FLOAT16);
    X4Data.update_input_desc_x(tensorDescX4);
    X4Data.update_output_desc_y(tensorDescX4);

    auto mul_op = op::Mul("Mul");
    mul_op.set_input_x1(add_op);
    mul_op.set_input_x2(X4Data);

    auto sigmoid_op = op::Sigmoid("Sigmoid");
    sigmoid_op.set_input_x(mul_op);

    auto mul_op_2 = op::Mul("Mul_2");
    mul_op_2.set_input_x1(sigmoid_op);
    mul_op_2.set_input_x2(add_op);

    // Additional test node
    auto X5Data = op::Data("x5");
    std::vector<int64_t> dims_x5{2048, 1981};
    ge::Shape shape_x5(dims_x5);
    ge::TensorDesc tensorDescX5(shape_x5, FORMAT_ND, DT_FLOAT16);
    X5Data.update_input_desc_x(tensorDescX5);
    X5Data.update_output_desc_y(tensorDescX5);

    auto mul_op_3 = op::Mul("Mul_3");
    mul_op_3.set_input_x1(mul_op_2);
    mul_op_3.set_input_x2(X5Data);

    std::vector<Operator> inputs{X1Data, X2Data, X3Data, X4Data, X5Data};
    std::vector<Operator> outputs{mul_op_3};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReshapeFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"Reshape", 3}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

TEST_F(batch_matmul_v2_reshape_fusion_test, batch_matmul_v2_reshape_fusion_test_8) {
    ge::Graph graph("batch_matmul_v2_reshape_fusion_test_8");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{-2};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{12, 3};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
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
    fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReshapeFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, false);
}

TEST_F(batch_matmul_v2_reshape_fusion_test, batch_matmul_v2_reshape_fusion_test_9)
{
    ge::Graph graph("batch_matmul_v2_reshape_fusion_test_9");
    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{4096, 16, 1775};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{1775, 1981};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);

    auto tanh_op = op::Tanh("tanh");
    tanh_op.set_input_x(bmOP);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{tanh_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReshapeFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"Reshape", 2}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

TEST_F(batch_matmul_v2_reshape_fusion_test, batch_matmul_v2_reshape_fusion_test_10)
{
    ge::Graph graph("batch_matmul_v2_reshape_fusion_test_10");
    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{4096, 16, 1775};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{1775, 1981};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);

    auto X3Data = op::Data("x3");
    std::vector<int64_t> dims_x3{1981};
    ge::Shape shape_x3(dims_x3);
    ge::TensorDesc tensorDescX3(shape_x3, FORMAT_ND, DT_FLOAT16);
    X3Data.update_input_desc_x(tensorDescX3);
    X3Data.update_output_desc_y(tensorDescX3);

    auto X4Data = op::Data("x4");
    std::vector<int64_t> dims_x4{1981};
    ge::Shape shape_x4(dims_x4);
    ge::TensorDesc tensorDescX4(shape_x4, FORMAT_ND, DT_FLOAT16);
    X4Data.update_input_desc_x(tensorDescX4);
    X4Data.update_output_desc_y(tensorDescX4);

    auto addn_op = op::AddN("AddN")
                       .create_dynamic_input_x(3)
                       .set_dynamic_input_x(0, bmOP)
                       .set_dynamic_input_x(1, X3Data)
                       .set_dynamic_input_x(2, X4Data)
                       .set_attr_N(3);

    std::vector<Operator> inputs{X1Data, X2Data, X3Data, X4Data};
    std::vector<Operator> outputs{addn_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReshapeFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"Reshape", 2}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}