#include <iostream>

#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include "image_ops.h"

using namespace ge;
using namespace op;

class grid_sampler2d_fusion_test : public testing::Test {
   protected:
    static void SetUpTestCase() { std::cout << "grid_sampler2d_fusion SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "grid_sampler2d_fusion TearDown" << std::endl; }
};

TEST_F(grid_sampler2d_fusion_test, grid_sampler2d_fusion_test_001) {
    ge::DataType dtype = ge::DT_FLOAT;
    ge::Format format = ge::FORMAT_ND;
    auto input_x = op::Data("x");
    std::vector<int64_t> x_dims{2, 3, 7, 5};
    ge::Shape x_shape(x_dims);
    ge::TensorDesc x_tensor_desc(x_shape, format, dtype);
    input_x.update_input_desc_x(x_tensor_desc);
    input_x.update_output_desc_y(x_tensor_desc);

    auto input_grid = op::Data("grid");
    std::vector<int64_t> grid_dims{2, 7, 5, 2};
    ge::Shape grid_shape(grid_dims);
    ge::TensorDesc grid_tensor_desc(grid_shape, format, dtype);
    input_grid.update_input_desc_x(grid_tensor_desc);
    input_grid.update_output_desc_y(grid_tensor_desc);

    auto grid_sampler2d_op = op::GridSampler2D("grid_sampler2d");
    grid_sampler2d_op.set_input_x_by_name(input_x, "input_x");
    grid_sampler2d_op.set_input_grid_by_name(input_grid, "input_grid");
    grid_sampler2d_op.set_attr_interpolation_mode("bilinear");
    grid_sampler2d_op.set_attr_padding_mode("zeros");
    grid_sampler2d_op.set_attr_align_corners(false);

    std::vector<Operator> inputs{input_x, input_grid};
    std::vector<Operator> outputs{grid_sampler2d_op};
    ge::Graph graph("grid_sampler2d_fusion_test_001");
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("GridSamplerFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool is_matched = true;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "GridUnnormal") {
            is_matched = true;
        }
    }
    EXPECT_EQ(is_matched, true);
}

TEST_F(grid_sampler2d_fusion_test, grid_sampler2d_fusion_test_002) {
    ge::DataType dtype = ge::DT_FLOAT;
    ge::Format format = ge::FORMAT_ND;
    auto input_x = op::Data("x");
    std::vector<int64_t> x_dims{2, 3, 7, 5};
    ge::Shape x_shape(x_dims);
    ge::TensorDesc x_tensor_desc(x_shape, format, dtype);
    input_x.update_input_desc_x(x_tensor_desc);
    input_x.update_output_desc_y(x_tensor_desc);

    auto input_grid = op::Data("grid");
    std::vector<int64_t> grid_dims{2, 7, 5, 2};
    ge::Shape grid_shape(grid_dims);
    ge::TensorDesc grid_tensor_desc(grid_shape, format, dtype);
    input_grid.update_input_desc_x(grid_tensor_desc);
    input_grid.update_output_desc_y(grid_tensor_desc);

    auto grid_sampler2d_op = op::GridSampler2D("grid_sampler2d");
    grid_sampler2d_op.set_input_x_by_name(input_x, "input_x");
    grid_sampler2d_op.set_input_grid_by_name(input_grid, "input_grid");
    grid_sampler2d_op.set_attr_interpolation_mode("bilinear");
    grid_sampler2d_op.set_attr_padding_mode("zeros");
    grid_sampler2d_op.set_attr_align_corners(false);

    std::vector<Operator> inputs{input_x, input_grid};
    std::vector<Operator> outputs{grid_sampler2d_op};
    ge::Graph graph("grid_sampler2d_fusion_test_002");
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("GridSamplerFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool is_matched = true;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "GridUnnormal") {
            is_matched = true;
        }
    }
    EXPECT_EQ(is_matched, true);
}