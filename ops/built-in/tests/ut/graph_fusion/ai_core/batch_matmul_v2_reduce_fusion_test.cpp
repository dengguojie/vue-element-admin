#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "matrix_calculation_ops.h"
#include "reduce_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "nonlinear_fuc_ops.h"

using namespace ge;
using namespace op;

class batch_matmul_v2_reduce_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "batch_matmul_v2_reduce_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "batch_matmul_v2_reduce_fusion_test TearDown" << std::endl;
    }
};

// BatchMatMulV2 --> ReduceSumD --> Output; b,m,k trans_a=false; b,k,n trans_b=false; k==1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_1)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_1");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 512, 1};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 1, 1024};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(false);
    bmOP.set_attr_adj_x2(false);

    std::vector<int64_t> dims_reduce_sum{512, 1024};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT16);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(bmOP);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape" || node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> Cast32 --> ReduceSumD --> Output; b,m,k trans_a=false; b,k,n trans_b=false; k==1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_2)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_2");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 512, 1};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 1, 1024};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(false);
    bmOP.set_attr_adj_x2(false);

    auto castOp = op::Cast("Cast_1");
    castOp.set_input_x(bmOP);
    castOp.set_attr_dst_type(ge::DT_FLOAT);

    std::vector<int64_t> dims_reduce_sum{512, 1024};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(castOp);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape" || node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> ReduceSumD --> Output; b,k,m trans_a=true; b,k,n trans_b=false; k==1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_3)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_3");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 1, 512};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 1, 1024};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(true);
    bmOP.set_attr_adj_x2(false);

    std::vector<int64_t> dims_reduce_sum{512, 1024};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT16);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(bmOP);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape" || node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> Cast32 --> ReduceSumD --> Output; b,k,m trans_a=true; b,k,n trans_b=false; k==1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_4)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_4");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 1, 512};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 1, 1024};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(true);
    bmOP.set_attr_adj_x2(false);

    auto castOp = op::Cast("Cast_1");
    castOp.set_input_x(bmOP);
    castOp.set_attr_dst_type(ge::DT_FLOAT);

    std::vector<int64_t> dims_reduce_sum{512, 1024};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(castOp);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape" || node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> ReduceSumD --> Output; b,m,k trans_a=false; b,n,k trans_b=true; k==1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_5)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_5");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 512, 1};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 1024, 1};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(false);
    bmOP.set_attr_adj_x2(true);

    std::vector<int64_t> dims_reduce_sum{512, 1024};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT16);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(bmOP);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape" || node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> Cast32 --> ReduceSumD --> Output; b,m,k trans_a=false; b,n,k trans_b=true; k==1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_6)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_6");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 512, 1};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 1024, 1};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(false);
    bmOP.set_attr_adj_x2(true);

    auto castOp = op::Cast("Cast_1");
    castOp.set_input_x(bmOP);
    castOp.set_attr_dst_type(ge::DT_FLOAT);

    std::vector<int64_t> dims_reduce_sum{512, 1024};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(castOp);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape" || node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> ReduceSumD --> Output; b,k,m trans_a=true; b,n,k trans_b=true; k==1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_7)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_7");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 1, 512};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 1024, 1};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(true);
    bmOP.set_attr_adj_x2(true);

    std::vector<int64_t> dims_reduce_sum{512, 1024};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT16);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(bmOP);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape" || node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> Cast32 --> ReduceSumD --> Output; b,k,m trans_a=true; b,n,k trans_b=true; k==1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_8)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_8");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 1, 512};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 1024, 1};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(true);
    bmOP.set_attr_adj_x2(true);

    auto castOp = op::Cast("Cast_1");
    castOp.set_input_x(bmOP);
    castOp.set_attr_dst_type(ge::DT_FLOAT);

    std::vector<int64_t> dims_reduce_sum{512, 1024};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(castOp);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape" || node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> ReduceSumD --> Output; b,m,k trans_a=false; b,k,n trans_b=false; k!=1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_9)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_9");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 35, 29};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 29, 64};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(false);
    bmOP.set_attr_adj_x2(false);

    std::vector<int64_t> dims_reduce_sum{35, 64};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT16);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(bmOP);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"TransposeD", 1}, {"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD" ||
            node->GetType() == "Reshape" ||
            node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> Cast32 --> ReduceSumD --> Output; b,m,k trans_a=false; b,k,n trans_b=false; k!=1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_10)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_10");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 35, 29};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 29, 64};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(false);
    bmOP.set_attr_adj_x2(false);

    auto castOp = op::Cast("Cast_1");
    castOp.set_input_x(bmOP);
    castOp.set_attr_dst_type(ge::DT_FLOAT);

    std::vector<int64_t> dims_reduce_sum{35, 64};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(castOp);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"TransposeD", 1}, {"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD" ||
            node->GetType() == "Reshape" ||
            node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> ReduceSumD --> Output; b,k,m trans_a=true; b,k,n trans_b=false; k!=1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_11)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_11");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 29, 35};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 29, 64};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(true);
    bmOP.set_attr_adj_x2(false);

    std::vector<int64_t> dims_reduce_sum{35, 64};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT16);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(bmOP);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape" || node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> Cast32 --> ReduceSumD --> Output; b,k,m trans_a=true; b,k,n trans_b=false; k!=1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_12)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_12");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 29, 35};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 29, 64};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(true);
    bmOP.set_attr_adj_x2(false);

    auto castOp = op::Cast("Cast_1");
    castOp.set_input_x(bmOP);
    castOp.set_attr_dst_type(ge::DT_FLOAT);

    std::vector<int64_t> dims_reduce_sum{35, 64};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(castOp);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Reshape" || node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> ReduceSumD --> Output; b,m,k trans_a=false; b,n,k trans_b=true; k!=1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_13)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_13");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 35, 29};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 64, 29};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(false);
    bmOP.set_attr_adj_x2(true);

    std::vector<int64_t> dims_reduce_sum{35, 64};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT16);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(bmOP);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"TransposeD", 2}, {"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD" ||
            node->GetType() == "Reshape" ||
            node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> Cast32 --> ReduceSumD --> Output; b,m,k trans_a=false; b,n,k trans_b=true; k!=1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_14)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_14");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 35, 29};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 64, 29};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(false);
    bmOP.set_attr_adj_x2(true);

    auto castOp = op::Cast("Cast_1");
    castOp.set_input_x(bmOP);
    castOp.set_attr_dst_type(ge::DT_FLOAT);

    std::vector<int64_t> dims_reduce_sum{35, 64};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(castOp);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"TransposeD", 2}, {"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD" ||
            node->GetType() == "Reshape" ||
            node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> ReduceSumD --> Output; b,k,m trans_a=true; b,n,k trans_b=true; k!=1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_15)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_15");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 29, 35};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 64, 29};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(true);
    bmOP.set_attr_adj_x2(true);

    std::vector<int64_t> dims_reduce_sum{35, 64};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT16);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(bmOP);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"TransposeD", 1}, {"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD" ||
            node->GetType() == "Reshape" ||
            node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> Cast32 --> ReduceSumD --> Output; b,k,m trans_a=true; b,n,k trans_b=true; k!=1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_16)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_16");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 29, 35};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{5120, 64, 29};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(true);
    bmOP.set_attr_adj_x2(true);

    auto castOp = op::Cast("Cast_1");
    castOp.set_input_x(bmOP);
    castOp.set_attr_dst_type(ge::DT_FLOAT);

    std::vector<int64_t> dims_reduce_sum{35, 64};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(castOp);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::SUCCESS);

    std::map<std::string, uint32_t> expected = {{"TransposeD", 1}, {"Reshape", 2}, {"BatchMatMulV2", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD" ||
            node->GetType() == "Reshape" ||
            node->GetType() == "BatchMatMulV2") {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

// BatchMatMulV2 --> ReduceSumD --> Output; b,m,k trans_a=false; b,k,n trans_b=false; b==1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_17)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_17");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{1, 512, 1};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{1, 1, 1024};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(false);
    bmOP.set_attr_adj_x2(false);

    std::vector<int64_t> dims_reduce_sum{512, 1024};
    ge::Shape shape_reduce_sum(dims_reduce_sum);
    ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT16);
    std::vector<int64_t> axes_value{0};
    auto reducesumd_op = op::ReduceSumD("ReduceSumD_1");
    reducesumd_op.set_input_x(bmOP);
    reducesumd_op.set_attr_axes(axes_value);
    reducesumd_op.set_attr_keep_dims(false);
    reducesumd_op.update_output_desc_y(desc_reduce_sum);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(reducesumd_op);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::NOT_CHANGED);
}

// BatchMatMulV2 --> ReduceSumD --> Output; b,m,k trans_a=false; b,k,n trans_b=false; b==1
TEST_F(batch_matmul_v2_reduce_fusion_test, batch_matmul_v2_reduce_fusion_test_18)
{
    ge::Graph graph("batch_matmul_v2_reduce_fusion_test_18");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{5120, 4, 256};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(tensorDescX1);
    X1Data.update_output_desc_y(tensorDescX1);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{128, 256};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2_1");
    bmOP.set_input_x1(X1Data);
    bmOP.set_input_x2(X2Data);
    bmOP.set_attr_adj_x1(false);
    bmOP.set_attr_adj_x2(true);

    // Additional test node
    auto sigmoid_op = op::Sigmoid("Sigmoid_1");
    sigmoid_op.set_input_x(bmOP);

    std::vector<Operator> inputs{X1Data, X2Data};
    std::vector<Operator> outputs{sigmoid_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulV2ReduceFusionPass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    ASSERT_EQ(ret, fe::NOT_CHANGED);
}
