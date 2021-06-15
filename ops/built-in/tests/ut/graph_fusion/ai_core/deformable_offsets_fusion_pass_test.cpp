#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class deformable_offsets_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "deformable_offsets_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "deformable_offsets_fusion_pass_test TearDown" << std::endl;
    }
};

TEST_F(deformable_offsets_fusion_pass_test, deformable_offsets_fusion_pass_test_1) {
    ge::Graph graph("deformable_offsets_fusion_pass_test_1");

    // input data x
    auto x_data = op::Data("x_data");
    std::vector<int64_t> dims_x{1, 304, 304, 256};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT);
    x_data.update_input_desc_x(tensorDescX);
    x_data.update_output_desc_y(tensorDescX);

    // input data offset
    auto offset_data = op::Data("offset_data");
    std::vector<int64_t> dims_offset{1, 152, 152, 27};
    ge::Shape shape_offset(dims_offset);
    ge::TensorDesc tensorDescOffset(shape_offset, FORMAT_NHWC, DT_FLOAT);
    offset_data.update_input_desc_x(tensorDescOffset);
    offset_data.update_output_desc_y(tensorDescOffset);

    // get input data
    auto deformable_op = op::DeformableOffsets("DeformableOffsets");
    deformable_op.set_input_x(x_data);
    deformable_op.set_input_offsets(offset_data);

    // create attr
    std::vector<int64_t> strides_attr = {1,2,2,1};
    std::vector<int64_t> pads_attr = {1,1,1,1};
    std::vector<int64_t> ksize_attr = {3,3};
    std::vector<int64_t> dilations_attr = {1,1,1,1};
    deformable_op.SetAttr("strides", strides_attr);
    deformable_op.SetAttr("pads",pads_attr);
    deformable_op.SetAttr("ksize", ksize_attr);
    deformable_op.SetAttr("dilations", dilations_attr);
    deformable_op.SetAttr("data_format","NHWC");
    deformable_op.SetAttr("deformable_groups", 1);
    deformable_op.SetAttr("modulated", true);


    //  fussion pass
    std::vector<Operator> inputs{x_data, offset_data};
    std::vector<Operator> outputs{deformable_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("DeformableOffsetsFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findDeformableOffsets = false;

    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DeformableOffsets") {
            findDeformableOffsets = true;
            break;
        }
    }
    EXPECT_EQ(findDeformableOffsets, true);
}


TEST_F(deformable_offsets_fusion_pass_test, deformable_offsets_fusion_pass_test_2) {
    ge::Graph graph("deformable_offsets_fusion_pass_test_2");

    // input data x
    auto x_data = op::Data("x_data");
    std::vector<int64_t> dims_x{1, 304, 304, 256};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT16);
    x_data.update_input_desc_x(tensorDescX);
    x_data.update_output_desc_y(tensorDescX);

    // input data offset
    auto offset_data = op::Data("offset_data");
    std::vector<int64_t> dims_offset{1, 152, 152, 27};
    ge::Shape shape_offset(dims_offset);
    ge::TensorDesc tensorDescOffset(shape_offset, FORMAT_NHWC, DT_FLOAT16);
    offset_data.update_input_desc_x(tensorDescOffset);
    offset_data.update_output_desc_y(tensorDescOffset);

    // get input data
    auto deformable_op = op::DeformableOffsets("DeformableOffsets");
    deformable_op.set_input_x(x_data);
    deformable_op.set_input_offsets(offset_data);

    // create attr
    std::vector<int64_t> strides_attr = {1,2,2,1};
    std::vector<int64_t> pads_attr = {1,1,1,1};
    std::vector<int64_t> ksize_attr = {3,3};
    std::vector<int64_t> dilations_attr = {1,1,1,1};
    deformable_op.SetAttr("strides", strides_attr);
    deformable_op.SetAttr("pads",pads_attr);
    deformable_op.SetAttr("ksize", ksize_attr);
    deformable_op.SetAttr("dilations", dilations_attr);
    deformable_op.SetAttr("data_format","NHWC");
    deformable_op.SetAttr("deformable_groups", 1);
    deformable_op.SetAttr("modulated", true);


    //  fussion pass
    std::vector<Operator> inputs{x_data, offset_data};
    std::vector<Operator> outputs{deformable_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("DeformableOffsetsFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findDeformableOffsets = false;

    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DeformableOffsets") {
            findDeformableOffsets = true;
            break;
        }
    }
    EXPECT_EQ(findDeformableOffsets, true);
}


TEST_F(deformable_offsets_fusion_pass_test, deformable_offsets_fusion_pass_test_3) {
    ge::Graph graph("deformable_offsets_fusion_pass_test_3");

    // input data x
    auto x_data = op::Data("x_data");
    std::vector<int64_t> dims_x{1, 304, 304, 256};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW, DT_FLOAT);
    x_data.update_input_desc_x(tensorDescX);
    x_data.update_output_desc_y(tensorDescX);

    // input data offset
    auto offset_data = op::Data("offset_data");
    std::vector<int64_t> dims_offset{1, 152, 152, 27};
    ge::Shape shape_offset(dims_offset);
    ge::TensorDesc tensorDescOffset(shape_offset, FORMAT_NCHW,  DT_FLOAT);
    offset_data.update_input_desc_x(tensorDescOffset);
    offset_data.update_output_desc_y(tensorDescOffset);

    // get input data
    auto deformable_op = op::DeformableOffsets("DeformableOffsets");
    deformable_op.set_input_x(x_data);
    deformable_op.set_input_offsets(offset_data);

    // create attr
    std::vector<int64_t> strides_attr = {1,2,2,1};
    std::vector<int64_t> pads_attr = {1,1,1,1};
    std::vector<int64_t> ksize_attr = {3,3};
    std::vector<int64_t> dilations_attr = {1,1,1,1};
    deformable_op.SetAttr("strides", strides_attr);
    deformable_op.SetAttr("pads",pads_attr);
    deformable_op.SetAttr("ksize", ksize_attr);
    deformable_op.SetAttr("dilations", dilations_attr);
    deformable_op.SetAttr("data_format","NHWC");
    deformable_op.SetAttr("deformable_groups", 1);
    deformable_op.SetAttr("modulated", true);


    //  fussion pass
    std::vector<Operator> inputs{x_data, offset_data};
    std::vector<Operator> outputs{deformable_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("DeformableOffsetsFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findDeformableOffsets = false;

    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DeformableOffsets") {
            findDeformableOffsets = true;
            break;
        }
    }
    EXPECT_EQ(findDeformableOffsets, true);
}