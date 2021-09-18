#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include <iostream>

using namespace ge;
using namespace op;

class cast_remove_fusion_test : public testing::Test {
  protected:
    static void SetUpTestCase()
    {
        std::cout << "cast_remove_fusion SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "cast_remove_fusion TearDown" << std::endl;
    }
};

TEST_F(cast_remove_fusion_test, cast_remove_fusion_test_1) {
    ge::Graph graph("cast_remove_fusion_test_1");

    auto input_x = op::Data("input_x");

    std::vector<int64_t> dims_x{1, 32, 32, 32};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescData(shape_x, ge::FORMAT_NCHW, ge::DT_BOOL);
    input_x.update_input_desc_x(tensorDescData);
    input_x.update_output_desc_y(tensorDescData);

    auto cast1_op = op::Cast("cast_1");
    ge::TensorDesc tensorDescCast1Out(shape_x, ge::FORMAT_NCHW, ge::DT_BOOL);
    cast1_op.update_input_desc_x(tensorDescData);
    cast1_op.update_output_desc_y(tensorDescCast1Out);

    cast1_op.set_input_x(input_x)
        .set_attr_dst_type(12);

    std::vector<Operator> inputs{input_x, cast1_op};
    std::vector<Operator> outputs{cast1_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "cast_remove_fusion_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("CastRemoveFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "cast_remove_fusion_test_1_after");
    bool findFusionCast = false;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Cast") {
            findFusionCast = true;
        }
    }
    EXPECT_EQ(findFusionCast, false);
}

TEST_F(cast_remove_fusion_test, cast_remove_fusion_test_2) {
    ge::Graph graph("cast_remove_fusion_test_2");

    auto input_x = op::Data("input_x");

    std::vector<int64_t> dims_x{1, 32, 32, 32};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescData(shape_x, ge::FORMAT_NCHW, ge::DT_INT32);
    input_x.update_input_desc_x(tensorDescData);
    input_x.update_output_desc_y(tensorDescData);

    auto cast1_op = op::Cast("cast_1");
    ge::TensorDesc tensorDescCast1Out(shape_x, ge::FORMAT_NCHW, ge::DT_BOOL);
    cast1_op.update_input_desc_x(tensorDescData);
    cast1_op.update_output_desc_y(tensorDescCast1Out);

    cast1_op.set_input_x(input_x)
        .set_attr_dst_type(12);

    std::vector<Operator> inputs{input_x, cast1_op};
    std::vector<Operator> outputs{cast1_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "cast_remove_fusion_test_2_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("CastRemoveFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "cast_remove_test_2_after");
    bool findFusionCast = false;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Cast") {
            findFusionCast = true;
        }
    }
    EXPECT_EQ(findFusionCast, true);
}

TEST_F(cast_remove_fusion_test, cast_remove_fusion_test_3) {
    OpDescPtr data = std::make_shared<OpDesc>("data0", "Data");
    OpDescPtr cast_op = std::make_shared<OpDesc>("cast", "Cast");
    OpDescPtr exp_op = std::make_shared<OpDesc>("exp", "Exp");

    vector<int64_t> input_dim = {10, 3, 32, 32};
    GeShape input_shape(input_dim);
    GeTensorDesc input_tenosr_desc(input_shape, FORMAT_NCHW, DT_FLOAT);
    input_tenosr_desc.SetOriginFormat(FORMAT_NCHW);
    input_tenosr_desc.SetOriginDataType(DT_FLOAT);
    input_tenosr_desc.SetOriginShape(input_shape);

    vector<int64_t> output_dim = {10, 3, 32, 32};
    GeShape output_shape(output_dim);
    GeTensorDesc output_tenosr_desc(output_shape, FORMAT_NCHW, DT_FLOAT);
    output_tenosr_desc.SetOriginFormat(FORMAT_NCHW);
    output_tenosr_desc.SetOriginDataType(DT_FLOAT);
    output_tenosr_desc.SetOriginShape(output_shape);

    data->AddOutputDesc(input_tenosr_desc);
    cast_op->AddInputDesc("x", input_tenosr_desc);
    cast_op->AddOutputDesc(output_tenosr_desc);
    exp_op->AddInputDesc("x", input_tenosr_desc);
    exp_op->AddOutputDesc(output_tenosr_desc);

    ge::AttrUtils::SetInt(cast_op, "dst_type", 0);

    ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ComputeGraph>("control anchor test");
    NodePtr data_node = compute_graph_ptr->AddNode(data);
    NodePtr cast_node = compute_graph_ptr->AddNode(cast_op);
    NodePtr exp_node = compute_graph_ptr->AddNode(exp_op);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), exp_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data_node->GetOutControlAnchor(), cast_node->GetInControlAnchor());

    fe::FusionPassTestUtils::RunGraphFusionPass("CastRemoveFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findFusionCast = false;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Cast") {
            findFusionCast = true;
        }
    }
    EXPECT_EQ(findFusionCast, false);
}