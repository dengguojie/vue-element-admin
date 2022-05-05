#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "rnn.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class resize_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "resize_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "resize_fusion_test TearDown" << std::endl;
    }
};

TEST_F(resize_fusion_test, resize_fusion_test_1) {
    ge::Graph graph("resize_fusion_test");

    auto xData = op::Data("xData");
    std::vector<int64_t> dims_x{16,16};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND, DT_FLOAT);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    auto roiData = op::Data("roiData");
    std::vector<int64_t> dims_roi{1,};
    ge::Shape shape_roi(dims_roi);
    ge::TensorDesc tensorDescRoi(shape_roi, FORMAT_ND, DT_FLOAT);
    roiData.update_input_desc_x(tensorDescRoi);
    roiData.update_output_desc_y(tensorDescRoi);
    
    auto scalesData = op::Data("scalesData");
    std::vector<int64_t> dims_scales{2,};
    ge::Shape shape_scales(dims_scales);
    ge::TensorDesc tensorDescScales(shape_scales, FORMAT_ND, DT_FLOAT);
    scalesData.update_input_desc_x(tensorDescScales);
    scalesData.update_output_desc_y(tensorDescScales);
    
    auto resizeOp = op::Resize("resize");
    resizeOp.set_input_x(xData)
            .set_input_roi(roiData)
            .set_input_scales(scalesData)
            .set_attr_coordinate_transformation_mode("half_pixel")
            .set_attr_cubic_coeff_a(-0.75)
            .set_attr_exclude_outside(0)
            .set_attr_extrapolation_value(0.0)
            .set_attr_mode("nearest")
            .set_attr_nearest_mode("round_prefer_floor");

    std::vector<Operator> inputs{xData, roiData, scalesData};
    std::vector<Operator> outputs{resizeOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ResizeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
}

TEST_F(resize_fusion_test, resize_fusion_test_2) {
    ge::Graph graph("resize_fusion_test");

    auto xData = op::Data("xData");
    std::vector<int64_t> dims_x{16,16};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND, DT_FLOAT);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    auto roiData = op::Data("roiData");
    std::vector<int64_t> dims_roi{1,};
    ge::Shape shape_roi(dims_roi);
    ge::TensorDesc tensorDescRoi(shape_roi, FORMAT_ND, DT_FLOAT);
    roiData.update_input_desc_x(tensorDescRoi);
    roiData.update_output_desc_y(tensorDescRoi);
    
    auto scalesData = op::Data("scalesData");
    std::vector<int64_t> dims_scales{2,};
    ge::Shape shape_scales(dims_scales);
    ge::TensorDesc tensorDescScales(shape_scales, FORMAT_ND, DT_FLOAT);
    scalesData.update_input_desc_x(tensorDescScales);
    scalesData.update_output_desc_y(tensorDescScales);
    
    auto resizeOp = op::Resize("resize");
    resizeOp.set_input_x(xData)
            .set_input_roi(roiData)
            .set_input_scales(scalesData)
            .set_attr_coordinate_transformation_mode("half_pixel")
            .set_attr_cubic_coeff_a(-0.75)
            .set_attr_exclude_outside(0)
            .set_attr_extrapolation_value(0.0)
            .set_attr_mode("nearest")
            .set_attr_nearest_mode("round_prefer_floor");

    std::vector<Operator> inputs{xData, roiData, scalesData};
    std::vector<Operator> outputs{resizeOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ResizeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
}
