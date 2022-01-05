#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "nn_detect_ops.h"


using namespace ge;
using namespace op;

class yolo_v2_detection_output_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "yolo_v2_detection_output_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "yolo_v2_detection_output_fusion_test TearDown" << std::endl;
    }
};

TEST_F(yolo_v2_detection_output_fusion_test, yolo_v2_detection_output_fusion_test_1) {
    ge::Graph graph("yolo_v2_detection_output_fusion_test_1");
    auto coord_datadata = op::Data("coord_data");
    auto obj_probdata = op::Data("obj_prob");
    auto classes_probdata = op::Data("classes_prob");
    auto img_infodata = op::Data("img_info");

    std::vector<int64_t> dims{10, 10, 10, 10, 10};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc0(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    coord_datadata.update_input_desc_x(tensorDesc0);
    coord_datadata.update_output_desc_y(tensorDesc0);

    std::vector<int64_t> dims1{10, 10, 10, 10, 10};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT);
    obj_probdata.update_input_desc_x(tensorDesc1);
    obj_probdata.update_output_desc_y(tensorDesc1);

    std::vector<int64_t> dims2{10, 10, 10, 10, 10};
    ge::Shape shape2(dims2);
    ge::TensorDesc tensorDesc2(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
    classes_probdata.update_input_desc_x(tensorDesc2);
    classes_probdata.update_output_desc_y(tensorDesc2);

    std::vector<int64_t> dims3{10, 10, 10, 10, 10};
    ge::Shape shape3(dims3);
    ge::TensorDesc tensorDesc3(shape3, ge::FORMAT_ND, ge::DT_FLOAT);
    img_infodata.update_input_desc_x(tensorDesc3);
    img_infodata.update_output_desc_y(tensorDesc3);

    auto yoloV2DetectionOutput_op = op::YoloV2DetectionOutput("YoloV2DetectionOutput1");
    yoloV2DetectionOutput_op.set_input_coord_data(coord_datadata)
                            .set_input_obj_prob(obj_probdata)
                            .set_input_classes_prob(classes_probdata)
                            .set_input_img_info(img_infodata)
                            .set_attr_biases({1.0,1.0,1.0,1.0,1.0});
    std::vector<Operator> inputs{coord_datadata,obj_probdata,classes_probdata,img_infodata};
    std::vector<Operator> outputs{yoloV2DetectionOutput_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("YoloV2DetectionOutputPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findYoloV2DetectionOutputD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "YoloV2DetectionOutputD") {
            findYoloV2DetectionOutputD = true;
        }
    }
    EXPECT_EQ(findYoloV2DetectionOutputD, true);
}

TEST_F(yolo_v2_detection_output_fusion_test, yolo_v2_detection_output_fusion_test_2) {
    ge::Graph graph("yolo_v2_detection_output_fusion_test_2");
    auto coord_datadata = op::Data("coord_data");
    auto obj_probdata = op::Data("obj_prob");
    auto classes_probdata = op::Data("classes_prob");
    auto img_infodata = op::Data("img_info");

    std::vector<int64_t> dims{-1, -1, -1, -1, -1};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc0(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    coord_datadata.update_input_desc_x(tensorDesc0);
    coord_datadata.update_output_desc_y(tensorDesc0);

    std::vector<int64_t> dims1{10, 10, 10, 10, 10};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT);
    obj_probdata.update_input_desc_x(tensorDesc1);
    obj_probdata.update_output_desc_y(tensorDesc1);

    std::vector<int64_t> dims2{10, 10, 10, 10, 10};
    ge::Shape shape2(dims2);
    ge::TensorDesc tensorDesc2(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
    classes_probdata.update_input_desc_x(tensorDesc2);
    classes_probdata.update_output_desc_y(tensorDesc2);

    std::vector<int64_t> dims3{10, 10, 10, 10, 10};
    ge::Shape shape3(dims3);
    ge::TensorDesc tensorDesc3(shape3, ge::FORMAT_ND, ge::DT_FLOAT);
    img_infodata.update_input_desc_x(tensorDesc3);
    img_infodata.update_output_desc_y(tensorDesc3);

    auto yoloV2DetectionOutput_op = op::YoloV2DetectionOutput("YoloV2DetectionOutput2");
    yoloV2DetectionOutput_op.set_input_coord_data(coord_datadata)
                            .set_input_obj_prob(obj_probdata)
                            .set_input_classes_prob(classes_probdata)
                            .set_input_img_info(img_infodata)
                            .set_attr_biases({1.0,1.0,1.0,1.0,1.0});
    std::vector<Operator> inputs{coord_datadata,obj_probdata,classes_probdata,img_infodata};
    std::vector<Operator> outputs{yoloV2DetectionOutput_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("YoloV2DetectionOutputPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findYoloV2DetectionOutputD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "YoloV2DetectionOutputD") {
            findYoloV2DetectionOutputD = true;
        }
    }
    EXPECT_EQ(findYoloV2DetectionOutputD, false);
}