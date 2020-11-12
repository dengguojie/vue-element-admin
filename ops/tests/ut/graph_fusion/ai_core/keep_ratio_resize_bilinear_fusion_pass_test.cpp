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

class keep_ratio_resize_bilinear_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "keep_ratio_resize_bilinear_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "keep_ratio_resize_bilinear_fusion_test TearDown" << std::endl;
    }
};

TEST_F(keep_ratio_resize_bilinear_fusion_test, keep_ratio_resize_bilinear_fusion_test_1) {
    ge::Graph graph("keep_ratio_resize_bilinear_fusion_test_1");

    auto imageData = op::Data("imageData");
    std::vector<int64_t> dims_x{2, 3096, 3096, 3};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_FLOAT16);
    imageData.update_input_desc_x(tensorDescX);
    imageData.update_output_desc_y(tensorDescX);

    auto keepRationOp = op::KeepRatioResizeBilinear("KeepRatioResizeBilinear_1");
    keepRationOp.set_input_images(imageData);
    keepRationOp.set_attr_min_dimension(3000);
    keepRationOp.set_attr_max_dimension(3000);

    std::vector<Operator> inputs{imageData};
    std::vector<Operator> outputs{keepRationOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

    fe::FusionPassTestUtils::RunGraphFusionPass("KeepRatioResizeBilinearFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr, false);

    bool findResizeBilinear = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ResizeBilinearV2") {
            findResizeBilinear = true;
        }
        if (node->GetType() == "ResizeBilinearV2D") {
            std::cout << "keep_ratio_resize_bilinear_fusion_test 33333" << std::endl;
        }

    }
    EXPECT_EQ(findResizeBilinear, true);
}

TEST_F(keep_ratio_resize_bilinear_fusion_test, keep_ratio_resize_bilinear_fusion_test_2) {
    ge::Graph graph("keep_ratio_resize_bilinear_fusion_test_2");

    auto imageData = op::Data("imageData");
    std::vector<int64_t> dims_x{2, 600, 1024, 3};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_FLOAT16);
    imageData.update_input_desc_x(tensorDescX);
    imageData.update_output_desc_y(tensorDescX);

    auto keepRationOp = op::KeepRatioResizeBilinear("KeepRatioResizeBilinear_2");
    keepRationOp.set_input_images(imageData);
    keepRationOp.set_attr_min_dimension(600);
    keepRationOp.set_attr_max_dimension(1024);

    std::vector<Operator> inputs{imageData};
    std::vector<Operator> outputs{keepRationOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("KeepRatioResizeBilinearFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr, true);

    bool findResizeBilinear = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ResizeBilinearV2D") {
            findResizeBilinear = true;
        }
    }
    EXPECT_EQ(findResizeBilinear, true);
}

