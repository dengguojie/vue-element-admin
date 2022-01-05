#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "state_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
using namespace op;

class mul_addn_l2loss_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "mul_addn_l2loss_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "mul_addn_l2loss_fusion_test TearDown" << std::endl;
    }
};

// TestCase_1 has cycle
TEST_F(mul_addn_l2loss_fusion_test, mul_addn_l2loss_fusion_test_1) {
    Graph graph("mul_addn_l2loss_fusion_test_1");
    auto input_data_1 = op::Data("input_data_1");
    auto input_data_2 = op::Data("input_data_2");

    auto dims = std::vector<int64_t>({1});
    ge::Shape shape(dims);
    TensorDesc tensorDesc(shape, FORMAT_ND, DT_FLOAT16);

    input_data_1.update_input_desc_x(tensorDesc);
    input_data_1.update_output_desc_y(tensorDesc);
    input_data_2.update_input_desc_x(tensorDesc);
    input_data_2.update_output_desc_y(tensorDesc);

    auto var_l2loss_mul_input = op::Variable("var_l2loss_mul_input");
    var_l2loss_mul_input.update_output_desc_y(tensorDesc);
    auto addn_op_1 = op::AddN("AddN_1")
                      .create_dynamic_input_x(1)
                      .set_dynamic_input_x(0, var_l2loss_mul_input)
                      .set_attr_N(1);
    auto mul_op_1 = op::Mul("Mul_1")
                      .set_input_x1(addn_op_1)
                      .set_input_x2(input_data_2);
    auto mul_op_2 = op::Mul("Mul_2")
                      .set_input_x1(var_l2loss_mul_input)
                      .set_input_x2(input_data_1);
    auto addn_op_2 = op::AddN("AddN_2")
                      .create_dynamic_input_x(2)
                      .set_dynamic_input_x(0, mul_op_2)
                      .set_dynamic_input_x(1, mul_op_1)
                      .set_attr_N(2);
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(addn_op_2);
    std::vector<Operator> inputs{input_data_1, input_data_2};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ComputeGraphPtr compute_graph_ptr = GraphUtils::GetComputeGraph(graph);

    // after generate the graph, add l2loss to the graph
    OpDescPtr l2_loss_op = std::make_shared<OpDesc>("l2loss", "L2Loss");
    GeTensorDesc l2_loss_tensor_desc(GeShape({1}), FORMAT_ND, DT_FLOAT16);
    l2_loss_tensor_desc.SetOriginShape(GeShape({1}));
    l2_loss_tensor_desc.SetOriginFormat(FORMAT_ND);
    GeTensorDesc l2_loss_output_desc(GeShape({1}), FORMAT_ND, DT_FLOAT16);
    l2_loss_output_desc.SetOriginShape(GeShape({1}));
    l2_loss_output_desc.SetOriginFormat(FORMAT_ND);
    l2_loss_op->AddInputDesc(l2_loss_tensor_desc);
    l2_loss_op->AddOutputDesc(l2_loss_output_desc);
    auto l2loss_node = compute_graph_ptr->AddNode(l2_loss_op);
    // change Edge
    NodePtr src_node;
    NodePtr dst_node;
    for(auto &node : compute_graph_ptr->GetAllNodes()){
        if(node->GetName() == "var_l2loss_mul_input" ){
            src_node = node;
        }else if(node->GetName() == "AddN_1"){
            dst_node = node;
        }
    }
    GraphUtils::RemoveEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

    GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), l2loss_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(l2loss_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
    
    // add TopologicalSorting before fusion
    if(ge::GRAPH_SUCCESS != compute_graph_ptr->TopologicalSorting()){
        std::cout<<"TopologicalSorting failed!"<<std::endl;
    }
    
    fe::FusionPassTestUtils::RunGraphFusionPass("MulAddNL2LossFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool passMatch = false;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "FusedMulAddNL2loss") {
            passMatch = true;
        }
    }
    EXPECT_EQ(passMatch, false);
}

// TestCase_2 doesn't have cycle
TEST_F(mul_addn_l2loss_fusion_test, mul_addn_l2loss_fusion_test_2) {
    Graph graph("mul_addn_l2loss_fusion_test_2");
    auto input_data_1 = op::Data("input_data_1");
    auto input_data_2 = op::Data("input_data_2");
    auto input_data_3 = op::Data("input_data_3");
    

    auto dims = std::vector<int64_t>({1});
    ge::Shape shape(dims);
    TensorDesc tensorDesc(shape, FORMAT_ND, DT_FLOAT16);

    input_data_1.update_input_desc_x(tensorDesc);
    input_data_1.update_output_desc_y(tensorDesc);
    input_data_2.update_input_desc_x(tensorDesc);
    input_data_2.update_output_desc_y(tensorDesc);
    input_data_3.update_input_desc_x(tensorDesc);
    input_data_3.update_output_desc_y(tensorDesc);

    auto var_l2loss_mul_input = op::Variable("var_l2loss_mul_input");
    var_l2loss_mul_input.update_output_desc_y(tensorDesc);
    auto addn_op_1 = op::AddN("AddN_1")
                      .create_dynamic_input_x(1)
                      .set_dynamic_input_x(0, var_l2loss_mul_input)
                      .set_attr_N(1);
    auto mul_op_1 = op::Mul("Mul_1")
                      .set_input_x1(addn_op_1)
                      .set_input_x2(input_data_2);
    auto mul_op_2 = op::Mul("Mul_2")
                      .set_input_x1(var_l2loss_mul_input)
                      .set_input_x2(input_data_1);
    auto addn_op_2 = op::AddN("AddN_2")
                      .create_dynamic_input_x(2)
                      .set_dynamic_input_x(0, mul_op_2)
                      .set_dynamic_input_x(1, input_data_3)
                      .set_attr_N(2);
    auto end_op_0= op::Square("end_op_0");
    end_op_0.set_input_x(addn_op_2);
    auto end_op_1 = op::Square("end_op_1");
    end_op_1.set_input_x(mul_op_1);
    std::vector<Operator> inputs{input_data_1, input_data_2};
    std::vector<Operator> outputs{end_op_0, end_op_1};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ComputeGraphPtr compute_graph_ptr = GraphUtils::GetComputeGraph(graph);

    // after generate the graph, add l2loss to the graph
    OpDescPtr l2_loss_op = std::make_shared<OpDesc>("l2loss", "L2Loss");
    GeTensorDesc l2_loss_tensor_desc(GeShape({1}), FORMAT_ND, DT_FLOAT16);
    l2_loss_tensor_desc.SetOriginShape(GeShape({1}));
    l2_loss_tensor_desc.SetOriginFormat(FORMAT_ND);
    GeTensorDesc l2_loss_output_desc(GeShape({1}), FORMAT_ND, DT_FLOAT16);
    l2_loss_output_desc.SetOriginShape(GeShape({1}));
    l2_loss_output_desc.SetOriginFormat(FORMAT_ND);
    l2_loss_op->AddInputDesc(l2_loss_tensor_desc);
    l2_loss_op->AddOutputDesc(l2_loss_output_desc);
    auto l2loss_node = compute_graph_ptr->AddNode(l2_loss_op);
    // change Edge
    NodePtr src_node;
    NodePtr dst_node;
    for(auto &node : compute_graph_ptr->GetAllNodes()){
        if(node->GetName() == "var_l2loss_mul_input" ){
            src_node = node;
        }else if(node->GetName() == "AddN_1"){
            dst_node = node;
        }
    }
    GraphUtils::RemoveEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
    
    GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), l2loss_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(l2loss_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
    
    // add TopologicalSorting before fusion
    if(ge::GRAPH_SUCCESS != compute_graph_ptr->TopologicalSorting()){
        std::cout<<"TopologicalSorting failed!"<<std::endl;
    }

    fe::FusionPassTestUtils::RunGraphFusionPass("MulAddNL2LossFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool passMatch = false;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "FusedMulAddNL2loss") {
            passMatch = true;
        }
    }
    EXPECT_EQ(passMatch, true);
}