#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class realdiv_2_muls_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "realdiv_2_muls_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "realdiv_2_muls_fusion_test TearDown" << std::endl;
    }
};

TEST_F(realdiv_2_muls_fusion_test, realdiv_2_muls_fusion_test_1) {
    ge::Graph graph("realdiv_2_muls_fusion_test_1");

    auto input_x1 = op::Data("x1");
    std::vector<int64_t> dims_x{1, 32, 32, 32};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescData(shape_x, ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_x1.update_input_desc_x(tensorDescData);
    input_x1.update_output_desc_y(tensorDescData);

    auto input_x2 = op::Const("x2");
    ge::Tensor value;
    float* dataValue = new float[1];
    *dataValue = 2.0;
    value.SetTensorDesc(TensorDesc(ge::Shape({1, }), ge::FORMAT_NCHW, ge::DT_FLOAT));
    value.SetData((uint8_t*)dataValue, 4);
    ge::TensorDesc desc_data(ge::Shape({1, }), ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_x2.set_attr_value(value);
    input_x2.update_output_desc_y(desc_data);
    delete[] dataValue;

    auto realdiv_op = op::RealDiv("RealDiv");
    realdiv_op.update_input_desc_x1(tensorDescData);
    realdiv_op.update_output_desc_y(tensorDescData);
    
    realdiv_op.set_input_x1(input_x1)
              .set_input_x2(input_x2);

    std::vector<Operator> inputs{input_x1, input_x2, realdiv_op};
    std::vector<Operator> outputs{realdiv_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("RealDiv2MulsFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool findFusionMuls = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Muls") {
            findFusionMuls = true;
        }
    }
    EXPECT_EQ(findFusionMuls, true);
}

TEST_F(realdiv_2_muls_fusion_test, realdiv_2_muls_fusion_test_2) {
    ge::Graph graph("realdiv_2_muls_fusion_test_2");

    auto input_x1 = op::Data("x1");
    std::vector<int64_t> dims_x{1, 16, 32, 32};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescData(shape_x, ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_x1.update_input_desc_x(tensorDescData);
    input_x1.update_output_desc_y(tensorDescData);

    auto input_x2 = op::Data("x2");
    input_x2.update_input_desc_x(tensorDescData);
    input_x2.update_output_desc_y(tensorDescData);

    auto realdiv_op = op::RealDiv("RealDiv");
    realdiv_op.update_input_desc_x1(tensorDescData);
    realdiv_op.update_input_desc_x2(tensorDescData);
    realdiv_op.update_output_desc_y(tensorDescData);
    
    realdiv_op.set_input_x1(input_x1)
              .set_input_x2(input_x2);

    std::vector<Operator> inputs{input_x1, input_x2, realdiv_op};
    std::vector<Operator> outputs{realdiv_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("RealDiv2MulsFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool findFusionMuls = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Muls") {
            findFusionMuls = true;
        }
    }
    EXPECT_EQ(findFusionMuls, false);
}

TEST_F(realdiv_2_muls_fusion_test, realdiv_2_muls_fusion_test_3) {
    ge::Graph graph("realdiv_2_muls_fusion_test_3");

    auto input_x1 = op::Data("x1");
    std::vector<int64_t> dims_x{1, -1, 32, 32};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescData(shape_x, ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_x1.update_input_desc_x(tensorDescData);
    input_x1.update_output_desc_y(tensorDescData);

    auto input_x2 = op::Const("x2");
    Tensor value;
    float* dataValue = new float[1];
    *dataValue = 2.0;
    value.SetTensorDesc(TensorDesc(ge::Shape({1, }), ge::FORMAT_NCHW, ge::DT_FLOAT));
    value.SetData((uint8_t*)dataValue, 4);
    TensorDesc desc_data(ge::Shape({1, }), ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_x2.set_attr_value(value);
    input_x2.update_output_desc_y(desc_data);
    delete[] dataValue;

    auto realdiv_op = op::RealDiv("RealDiv");
    realdiv_op.update_input_desc_x1(tensorDescData);
    realdiv_op.update_output_desc_y(tensorDescData);
    
    realdiv_op.set_input_x1(input_x1)
              .set_input_x2(input_x2);

    std::vector<Operator> inputs{input_x1, input_x2, realdiv_op};
    std::vector<Operator> outputs{realdiv_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("RealDiv2MulsFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool findFusionMuls = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Muls") {
            findFusionMuls = true;
        }
    }
    EXPECT_EQ(findFusionMuls, false);
}

TEST_F(realdiv_2_muls_fusion_test, realdiv_2_muls_fusion_test_4) {
    ge::Graph graph("realdiv_2_muls_fusion_test_4");

    auto input_x1 = op::Data("x1");
    std::vector<int64_t> dims_x{1, 16, 32, 32};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescData(shape_x, ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_x1.update_input_desc_x(tensorDescData);
    input_x1.update_output_desc_y(tensorDescData);

    auto input_x2 = op::Const("x2");
    ge::Tensor value;
    float* dataValue = new float[2];
    *dataValue = 2.0;
    *(dataValue + 1) = 2.0;
    value.SetTensorDesc(TensorDesc(ge::Shape({1, }), ge::FORMAT_NCHW, ge::DT_FLOAT));
    value.SetData((uint8_t*)dataValue, 8);
    ge::TensorDesc desc_data(ge::Shape({1, }), ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_x2.set_attr_value(value);
    input_x2.update_output_desc_y(desc_data);
    delete[] dataValue;

    auto realdiv_op = op::RealDiv("RealDiv");
    realdiv_op.update_input_desc_x1(tensorDescData);
    realdiv_op.update_output_desc_y(tensorDescData);
    
    realdiv_op.set_input_x1(input_x1)
              .set_input_x2(input_x2);

    std::vector<Operator> inputs{input_x1, input_x2, realdiv_op};
    std::vector<Operator> outputs{realdiv_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("RealDiv2MulsFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool findFusionMuls = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Muls") {
            findFusionMuls = true;
        }
    }
    EXPECT_EQ(findFusionMuls, false);
}

TEST_F(realdiv_2_muls_fusion_test, realdiv_2_muls_fusion_test_5) {
    ge::Graph graph("realdiv_2_muls_fusion_test_5");

    auto input_x1 = op::Data("x1");
    std::vector<int64_t> dims_x{1, 16, 32, 32};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescData(shape_x, ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_x1.update_input_desc_x(tensorDescData);
    input_x1.update_output_desc_y(tensorDescData);

    auto input_x2 = op::Const("x2");
    ge::Tensor value;
    float* dataValue = new float[1];
    *dataValue = 0.0;
    value.SetTensorDesc(TensorDesc(ge::Shape({1, }), ge::FORMAT_NCHW, ge::DT_FLOAT));
    value.SetData((uint8_t*)dataValue, 4);
    ge::TensorDesc desc_data(ge::Shape({1, }), ge::FORMAT_NCHW, ge::DT_FLOAT);
    input_x2.set_attr_value(value);
    input_x2.update_output_desc_y(desc_data);
    delete[] dataValue;

    auto realdiv_op = op::RealDiv("RealDiv");
    realdiv_op.update_input_desc_x1(tensorDescData);
    realdiv_op.update_output_desc_y(tensorDescData);

    realdiv_op.set_input_x1(input_x1)
             .set_input_x2(input_x2);

    std::vector<Operator> inputs{input_x1, input_x2, realdiv_op};
    std::vector<Operator> outputs{realdiv_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("RealDiv2MulsFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool findFusionMuls = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Muls") {
            findFusionMuls = true;
        }
    }
    EXPECT_EQ(findFusionMuls, false);
}

TEST_F(realdiv_2_muls_fusion_test, realdiv_2_muls_fusion_test_6) {
    ge::Graph graph("realdiv_2_muls_fusion_test_6");

    OpDescPtr data = std::make_shared<OpDesc>("data0", "Data");
    OpDescPtr const_op = std::make_shared<OpDesc>("Const", "Const");
    OpDescPtr realdiv_op = std::make_shared<OpDesc>("realdiv", "RealDiv");

    std::vector<int64_t> dims_x{1, 16, 32, 32};
    ge::GeShape shape_x(dims_x);
    ge::GeTensorDesc tensorDescData(shape_x, ge::FORMAT_NCHW, ge::DT_FLOAT);
    data->AddOutputDesc(tensorDescData);

    ge::GeTensorDesc desc_data(ge::GeShape({1, }), ge::FORMAT_NCHW, ge::DT_FLOAT);
    const_op->AddOutputDesc(desc_data);

    realdiv_op->AddInputDesc("x1", tensorDescData);
    realdiv_op->AddInputDesc("x2", desc_data);
    realdiv_op->AddOutputDesc("y", tensorDescData);

    ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ComputeGraph>("control anchor test");
    NodePtr data_node = compute_graph_ptr->AddNode(data);
    NodePtr const_node = compute_graph_ptr->AddNode(const_op);
    NodePtr realdiv_node = compute_graph_ptr->AddNode(realdiv_op);

    float *const_value = new float[1];
    *const_value = 2.0;
    ge::GeTensorPtr weightTensor = nullptr;
    weightTensor = std::make_shared<GeTensor>(desc_data, reinterpret_cast<uint8_t *>(const_value), 4);
    ge::OpDescUtils::SetWeights(const_node, {weightTensor});

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), realdiv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), realdiv_node->GetInDataAnchor(1));

    GraphUtils::AddEdge(data_node->GetOutControlAnchor(), realdiv_node->GetInControlAnchor());

    fe::FusionPassTestUtils::RunGraphFusionPass("RealDiv2MulsFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findFusionMuls = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Muls") {
            findFusionMuls = true;
        }
    }
    EXPECT_EQ(findFusionMuls, true);
}
