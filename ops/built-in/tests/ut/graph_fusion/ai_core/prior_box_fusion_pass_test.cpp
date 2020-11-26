#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_detect_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class prior_box_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "inplace_add TearDown" << std::endl;
    }
};

// TEST_F(prior_box_fusion_pass_test, prior_box_fusion_pass_test_1) {
//     ge::Graph graph("prior_box_fusion_pass_test_1");
//     auto feature_input_data = op::Data("feature_input_data");
//     auto img_input_data = op::Data("img_input_data");
//     auto boxes_input_data = op::Data("boxes_input_data");
//     std::vector<int64_t> feature_dims{2, 16, 5, 5};
//     std::vector<int64_t> img_dims{2, 16, 300, 300};
//     std::vector<int64_t> boxes_dims{1, 2, 2400, 2400};
//     ge::Shape feature_shape(feature_dims);
//     ge::Shape img_shape(img_dims);
//     ge::Shape boxes_shape(boxes_dims);
//     ge::TensorDesc featureTensorDesc(feature_shape);
//     ge::TensorDesc imgTensorDesc(img_shape);
//     ge::TensorDesc boxesTensorDesc(boxes_shape);
//     feature_input_data.update_input_desc_x(featureTensorDesc);
//     feature_input_data.update_output_desc_y(featureTensorDesc);
//     img_input_data.update_input_desc_x(imgTensorDesc);
//     img_input_data.update_output_desc_y(imgTensorDesc);
//     boxes_input_data.update_input_desc_x(boxesTensorDesc);
//     boxes_input_data.update_output_desc_y(boxesTensorDesc);
// 
//     vector<float> min_size;
//     vector<float> max_size;
//     vector<float> asp;
//     vector<float> var;
//     min_size.push_back(162.0);
//     max_size.push_back(213.0);
//     asp.push_back(2);
//     asp.push_back(3);
//     var.push_back(0.1);
// 
//     auto prior_op = op::PriorBox("input_0");
//     prior_op.set_input_x(feature_input_data)
//             .set_input_img(img_input_data)
//             //.set_input_boxes(boxes_input_data)
//             .set_attr_min_size(min_size)
//             .set_attr_max_size(max_size)
//             .set_attr_aspect_ratio(asp)
//             .set_attr_flip(true)
//             .set_attr_clip(false)
//             .set_attr_variance(var)
//             .set_attr_img_h(300)
//             .set_attr_img_w(300)
//             .set_attr_step_h(64)
//             .set_attr_step_w(64)
//             .set_attr_offset(0.5);
// 
//     std::vector<Operator> inputs{feature_input_data, img_input_data, boxes_input_data};
//     std::vector<Operator> outputs{prior_op};
//     graph.SetInputs(inputs).SetOutputs(outputs);
//     ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
//     fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
//     //    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_before");
//     fe::FusionPassTestUtils::RunGraphFusionPass("PriorBoxPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
//     //    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_after");
// 
//     bool findNode = false;
//     bool shapeMatch = false;
//     //vector<int64_t> expectShape{3, 32, 3, 32};
//     for (auto node: compute_graph_ptr->GetAllNodes()) {
//         if (node->GetType() == "PriorBoxDV2") {
//             findNode = true;
//             auto inputDesc = node->GetOpDesc()->GetInputDesc(2);
//             std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
//             if (dims == boxes_dims) {
//                 shapeMatch = true;
//             }
//         }
//     }
//     EXPECT_EQ(findNode, true);
//     //EXPECT_EQ(shapeMatch, true);
// }
// 
// TEST_F(prior_box_fusion_pass_test, prior_box_fusion_pass_test_2) {
//     ge::Graph graph("prior_box_fusion_pass_test_2");
//     auto feature_input_data = op::Data("feature_input_data");
//     auto img_input_data = op::Data("img_input_data");
//     auto boxes_input_data = op::Data("boxes_input_data");
//     std::vector<int64_t> feature_dims{2, 16, 5, 5, 1};
//     std::vector<int64_t> img_dims{2, 16, 300, 300, 1};
//     std::vector<int64_t> boxes_dims{1, 2, 2400, 2400};
//     ge::Shape feature_shape(feature_dims);
//     ge::Shape img_shape(img_dims);
//     ge::Shape boxes_shape(boxes_dims);
//     ge::TensorDesc featureTensorDesc(feature_shape);
//     ge::TensorDesc imgTensorDesc(img_shape);
//     ge::TensorDesc boxesTensorDesc(boxes_shape);
//     feature_input_data.update_input_desc_x(featureTensorDesc);
//     feature_input_data.update_output_desc_y(featureTensorDesc);
//     img_input_data.update_input_desc_x(imgTensorDesc);
//     img_input_data.update_output_desc_y(imgTensorDesc);
//     boxes_input_data.update_input_desc_x(boxesTensorDesc);
//     boxes_input_data.update_output_desc_y(boxesTensorDesc);
// 
//     vector<float> min_size;
//     vector<float> max_size;
//     vector<float> asp;
//     vector<float> var;
//     min_size.push_back(162.0);
//     max_size.push_back(213.0);
//     asp.push_back(1.00000001);
//     asp.push_back(1.00000021);
//     var.push_back(0.1);
//     var.push_back(0.1);
//     var.push_back(0.1);
//     var.push_back(0.1);
// 
//     auto prior_op = op::PriorBox("input_0");
//     prior_op.set_input_x(feature_input_data)
//             .set_input_img(img_input_data)
//             //.set_input_boxes(boxes_input_data)
//             .set_attr_min_size(min_size)
//             .set_attr_max_size(max_size)
//             .set_attr_aspect_ratio(asp)
//             .set_attr_flip(true)
//             .set_attr_clip(true)
//             .set_attr_variance(var)
//             .set_attr_img_h(0)
//             .set_attr_img_w(0)
//             .set_attr_step_h(0)
//             .set_attr_step_w(0)
//             .set_attr_offset(0.5);
// 
//     std::vector<Operator> inputs{feature_input_data, img_input_data, boxes_input_data};
//     std::vector<Operator> outputs{prior_op};
//     graph.SetInputs(inputs).SetOutputs(outputs);
//     ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
//     fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
//     //    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_before");
//     fe::FusionPassTestUtils::RunGraphFusionPass("PriorBoxPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
//     //    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_after");
// 
//     bool findNode = false;
//     bool shapeMatch = false;
//     //vector<int64_t> expectShape{3, 32, 3, 32};
//     for (auto node: compute_graph_ptr->GetAllNodes()) {
//         if (node->GetType() == "PriorBoxDV2") {
//             findNode = true;
//             auto inputDesc = node->GetOpDesc()->GetInputDesc(2);
//             std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
//             if (dims == boxes_dims) {
//                 shapeMatch = true;
//             }
//         }
//     }
//     EXPECT_EQ(findNode, true);
//     //EXPECT_EQ(shapeMatch, true);
// }
