#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "image_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class Resize_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "Resize_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "Resize_fusion_test TearDown" << std::endl;
    }
};

TEST_F(Resize_fusion_test, Resize_fusion_test_1) {
    ge::Graph graph("Resize_fusion_test_1");

    auto imageData = op::Data("imageData");
    bool const_input = false;
    std::string mode_name = "nearest";
    std::vector<int64_t> dims_x{1, 1, 2, 2};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_FLOAT16);
    imageData.update_input_desc_x(tensorDescX);
    imageData.update_output_desc_y(tensorDescX);

    auto resizeop = op::Resize("resize");
    resizeop.set_input_x(imageData);
    resizeop.set_input_roi(imageData);
    resizeop.set_input_scales(imageData);
    resizeop.set_input_sizes(imageData);

    std::vector<Operator> inputs{imageData, imageData, imageData, imageData};
    std::vector<Operator> outputs{resizeop};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

    fe::FusionPassTestUtils::RunGraphFusionPass("ResizeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findResizeBilinear = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ResizeNearestNeighborV2") {
            findResizeBilinear = true;
        }
    }
    EXPECT_EQ(findResizeBilinear, true);
}

TEST_F(Resize_fusion_test, Resize_fusion_test_2) {
    ge::Graph graph("Resize_fusion_test_2");

    std::string mode_name = "linear";
    auto imageData = op::Data("imageData");
    bool const_input = false;
    std::vector<int64_t> dims_x{1, 1, 2, 2};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT16);
    imageData.update_input_desc_x(tensorDescX);
    imageData.update_output_desc_y(tensorDescX);

    auto resizeop = op::Resize("resize_2");
    resizeop.set_input_x(imageData);
    resizeop.set_input_roi(imageData);
    resizeop.set_input_scales(imageData);
    resizeop.set_input_sizes(imageData);

    std::vector<Operator> inputs{imageData, imageData, imageData, imageData};
    std::vector<Operator> outputs{resizeop};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ResizeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findResizeBilinear = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ResizeBilinearV2") {
            findResizeBilinear = true;
        }
    }
    EXPECT_EQ(findResizeBilinear, false);
}

TEST_F(Resize_fusion_test, Resize_fusion_test_4) {
    ge::Graph graph("Resize_fusion_test_4");

    auto imageData = op::Data("imageData");
    std::string mode_name = "nearest";
    bool const_input = true;
    std::vector<int64_t> dims_x{1, 1, 2, 2};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_FLOAT16);
    imageData.update_input_desc_x(tensorDescX);
    imageData.update_output_desc_y(tensorDescX);

    auto resizeop = op::Resize("resize");
    resizeop.set_input_x(imageData);
    resizeop.set_input_roi(imageData);
    resizeop.set_input_scales(imageData);

    std::vector<Operator> inputs{imageData,  imageData, imageData};
    std::vector<Operator> outputs{resizeop};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

    fe::FusionPassTestUtils::RunGraphFusionPass("ResizeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findResizeBilinear = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ResizeNearestNeighborV2D") {
            findResizeBilinear = true;
        }
    }
    EXPECT_EQ(findResizeBilinear, false);
}

TEST_F(Resize_fusion_test, Resize_fusion_test_5) {
    ge::Graph graph("Resize_fusion_test_5");

    auto imageData = op::Data("imageData");
    std::string mode_name = "linear";
    bool const_input = true;
    std::vector<int64_t> dims_x{1, 1, 2, 2};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT16);
    imageData.update_input_desc_x(tensorDescX);
    imageData.update_output_desc_y(tensorDescX);

    auto resizeop = op::Resize("resize_2");
    resizeop.set_input_x(imageData);
    resizeop.set_input_roi(imageData);
    resizeop.set_input_scales(imageData);

    std::vector<Operator> inputs{imageData, imageData, imageData};
    std::vector<Operator> outputs{resizeop};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ResizeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findResizeBilinear = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ResizeBilinearV2D") {
            findResizeBilinear = true;
        }
    }
    EXPECT_EQ(findResizeBilinear, false);
}

TEST_F(Resize_fusion_test, Resize_fusion_test_3) {
    ge::Graph graph("Resize_fusion_test_3");

    auto imageData = op::Data("imageData");
    bool const_input = true;
    std::string mode_name = "bilenr";
    std::vector<int64_t> dims_x{1};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_FLOAT16);
    imageData.update_input_desc_x(tensorDescX);
    imageData.update_output_desc_y(tensorDescX);

    auto resizeop = op::Resize("resize_2");
    resizeop.set_input_x(imageData);
    resizeop.set_input_sizes(imageData);

    std::vector<Operator> inputs{imageData, imageData};
    std::vector<Operator> outputs{resizeop};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ResizeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr, true);

    bool findResizeBilinear = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ResizeBilinearV2") {
            findResizeBilinear = true;
        }
    }
    EXPECT_EQ(findResizeBilinear, false);
}
