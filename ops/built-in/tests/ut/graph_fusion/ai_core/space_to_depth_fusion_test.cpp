#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class space_to_depth_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "space_to_depth_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "space_to_depth_fusion_test TearDown" << std::endl;
    }
};

TEST_F(space_to_depth_fusion_test, space_to_depth_fusion_test_1) {
    ge::Graph graph("space_to_depth_fusion_test_1");

    auto in_input_x_data = op::Data("diag_input_x");
    std::vector<int64_t> dims{2,4,4,2};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
	tensorDesc.SetOriginFormat(ge::FORMAT_NHWC); 
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto spacetodepth = op::SpaceToDepth("spacetodepth_0");
    spacetodepth.set_input_x(in_input_x_data)
        .set_attr_block_size({4});

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(spacetodepth);
	

    std::vector<Operator> inputs{in_input_x_data};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("SpaceToDepthFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findSpaceToDepth = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SpaceToDepth") {
            findSpaceToDepth = true;
        }
    }
    EXPECT_EQ(findSpaceToDepth, true);
}

TEST_F(space_to_depth_fusion_test, space_to_depth_fusion_test_2) {
    ge::Graph graph("space_to_depth_fusion_test_2");

    auto in_input_x_data = op::Data("diag_input_x");
    std::vector<int64_t> dims{1,24,36,1088};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
	tensorDesc.SetOriginFormat(ge::FORMAT_NHWC); 
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto spacetodepth = op::SpaceToDepth("spacetodepth_0");
    spacetodepth.set_input_x(in_input_x_data)
        .set_attr_block_size({6});

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(spacetodepth);
	

    std::vector<Operator> inputs{in_input_x_data};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("SpaceToDepthFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findSpaceToDepth = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SpaceToDepth") {
            findSpaceToDepth = true;
        }
    }
    EXPECT_EQ(findSpaceToDepth, true);
}

TEST_F(space_to_depth_fusion_test, space_to_depth_fusion_test_3) {
    ge::Graph graph("space_to_depth_fusion_test_3");

    auto in_input_x_data = op::Data("diag_input_x");
    std::vector<int64_t> dims{1,1088,24,36};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
	tensorDesc.SetOriginFormat(ge::FORMAT_NCHW); 
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto spacetodepth = op::SpaceToDepth("spacetodepth_0");
    spacetodepth.set_input_x(in_input_x_data)
        .set_attr_block_size({6});

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(spacetodepth);
	

    std::vector<Operator> inputs{in_input_x_data};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("SpaceToDepthFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findSpaceToDepth = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SpaceToDepth") {
            findSpaceToDepth = true;
        }
    }
    EXPECT_EQ(findSpaceToDepth, true);
}

TEST_F(space_to_depth_fusion_test, space_to_depth_fusion_test_4) {
    ge::Graph graph("space_to_depth_fusion_test_4");

    auto in_input_x_data = op::Data("diag_input_x");
    std::vector<int64_t> dims{4,4,2,2};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_HWCN, ge::DT_FLOAT16);
	tensorDesc.SetOriginFormat(ge::FORMAT_HWCN); 
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto spacetodepth = op::SpaceToDepth("spacetodepth_0");
    spacetodepth.set_input_x(in_input_x_data)
        .set_attr_block_size({4});

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(spacetodepth);
	

    std::vector<Operator> inputs{in_input_x_data};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("SpaceToDepthFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findSpaceToDepth = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SpaceToDepth") {
            findSpaceToDepth = true;
        }
    }
    EXPECT_EQ(findSpaceToDepth, true);
}

TEST_F(space_to_depth_fusion_test, space_to_depth_fusion_test_5) {
    ge::Graph graph("space_to_depth_fusion_test_5");

    auto in_input_x_data = op::Data("diag_input_x");
    std::vector<int64_t> dims{2,4,4,2};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT16);
	tensorDesc.SetOriginFormat(ge::FORMAT_ND); 
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto spacetodepth = op::SpaceToDepth("spacetodepth_0");
    spacetodepth.set_input_x(in_input_x_data)
        .set_attr_block_size({4});

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(spacetodepth);
	

    std::vector<Operator> inputs{in_input_x_data};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("SpaceToDepthFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findSpaceToDepth = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SpaceToDepth") {
            findSpaceToDepth = true;
        }
    }
    EXPECT_EQ(findSpaceToDepth, true);
}