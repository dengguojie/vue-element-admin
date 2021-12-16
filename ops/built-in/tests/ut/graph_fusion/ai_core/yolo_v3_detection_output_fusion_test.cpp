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

class yolo_v3_detection_output_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "yolo_v3_detection_output_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "yolo_v3_detection_output_fusion_test TearDown" << std::endl;
    }
};

TEST_F(yolo_v3_detection_output_fusion_test, yolo_v3_detection_output_fusion_test_1) {
    ge::Graph graph("yolo_v3_detection_output_fusion_test_1");
    auto coord_data_lowdata = op::Data("coord_data_low");
    auto coord_data_middata = op::Data("coord_data_mid");
    auto coord_data_highdata = op::Data("coord_data_high");
    auto obj_prob_lowdata = op::Data("obj_prob_low");
    auto obj_prob_middata = op::Data("obj_prob_mid");
    auto obj_prob_highdata = op::Data("obj_prob_high");
    auto classes_prob_lowdata = op::Data("classes_prob_low");
    auto classes_prob_middata = op::Data("classes_prob_mid");
    auto classes_prob_highdata = op::Data("classes_prob_high");
    auto img_infodata = op::Data("img_info");

    std::vector<int64_t> dims{10, 10, 10, 10, 10};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc0(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    coord_data_lowdata.update_input_desc_x(tensorDesc0);
    coord_data_lowdata.update_output_desc_y(tensorDesc0);

    std::vector<int64_t> dims1{10, 10, 10, 10, 10};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT);
    coord_data_middata.update_input_desc_x(tensorDesc1);
    coord_data_middata.update_output_desc_y(tensorDesc1);

    std::vector<int64_t> dims2{10, 10, 10, 10, 10};
    ge::Shape shape2(dims2);
    ge::TensorDesc tensorDesc2(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
    coord_data_highdata.update_input_desc_x(tensorDesc2);
    coord_data_highdata.update_output_desc_y(tensorDesc2);

    std::vector<int64_t> dims3{10, 10, 10, 10, 10};
    ge::Shape shape3(dims3);
    ge::TensorDesc tensorDesc3(shape3, ge::FORMAT_ND, ge::DT_FLOAT);
    obj_prob_lowdata.update_input_desc_x(tensorDesc3);
    obj_prob_lowdata.update_output_desc_y(tensorDesc3);

    std::vector<int64_t> dims4{10, 10, 10, 10, 10};
    ge::Shape shape4(dims4);
    ge::TensorDesc tensorDesc4(shape4, ge::FORMAT_ND, ge::DT_FLOAT);
    obj_prob_middata.update_input_desc_x(tensorDesc4);
    obj_prob_middata.update_output_desc_y(tensorDesc4);

    std::vector<int64_t> dims5{10, 10, 10, 10, 10};
    ge::Shape shape5(dims5);
    ge::TensorDesc tensorDesc5(shape5, ge::FORMAT_ND, ge::DT_FLOAT);
    obj_prob_highdata.update_input_desc_x(tensorDesc5);
    obj_prob_highdata.update_output_desc_y(tensorDesc5);

    std::vector<int64_t> dims6{10, 10, 10, 10, 10};
    ge::Shape shape6(dims6);
    ge::TensorDesc tensorDesc6(shape6, ge::FORMAT_ND, ge::DT_FLOAT);
    classes_prob_lowdata.update_input_desc_x(tensorDesc6);
    classes_prob_lowdata.update_output_desc_y(tensorDesc6);

    std::vector<int64_t> dims7{10, 10, 10, 10, 10};
    ge::Shape shape7(dims7);
    ge::TensorDesc tensorDesc7(shape7, ge::FORMAT_ND, ge::DT_FLOAT);
    classes_prob_middata.update_input_desc_x(tensorDesc7);
    classes_prob_middata.update_output_desc_y(tensorDesc7);

    std::vector<int64_t> dims8{10, 10, 10, 10, 10};
    ge::Shape shape8(dims8);
    ge::TensorDesc tensorDesc8(shape8, ge::FORMAT_ND, ge::DT_FLOAT);
    classes_prob_highdata.update_input_desc_x(tensorDesc8);
    classes_prob_highdata.update_output_desc_y(tensorDesc8);

    std::vector<int64_t> dims9{10, 10, 10, 10, 10};
    ge::Shape shape9(dims9);
    ge::TensorDesc tensorDesc9(shape9, ge::FORMAT_ND, ge::DT_FLOAT);
    img_infodata.update_input_desc_x(tensorDesc9);
    img_infodata.update_output_desc_y(tensorDesc9);

    auto yoloV3DetectionOutput_op = op::YoloV3DetectionOutput("YoloV3DetectionOutput1");
    yoloV3DetectionOutput_op.set_input_coord_data_low(coord_data_lowdata)
                            .set_input_coord_data_mid(coord_data_middata)
                            .set_input_coord_data_high(coord_data_highdata)
                            .set_input_obj_prob_low(obj_prob_lowdata)
                            .set_input_obj_prob_mid(obj_prob_middata)
                            .set_input_obj_prob_high(obj_prob_highdata)
                            .set_input_classes_prob_low(classes_prob_lowdata)
                            .set_input_classes_prob_mid(classes_prob_middata)
                            .set_input_classes_prob_high(classes_prob_highdata)
                            .set_input_img_info(img_infodata)
                            .set_attr_biases_low({1.0,1.0,1.0,1.0,1.0})
                            .set_attr_biases_mid({1.0,1.0,1.0,1.0,1.0})
                            .set_attr_biases_high({1.0,1.0,1.0,1.0,1.0});
    std::vector<Operator> inputs{coord_data_lowdata,coord_data_middata,coord_data_highdata,obj_prob_lowdata,obj_prob_middata,obj_prob_highdata,classes_prob_lowdata,classes_prob_middata,classes_prob_highdata,img_infodata};
    std::vector<Operator> outputs{yoloV3DetectionOutput_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("YoloV3DetectionOutputPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findYoloV3DetectionOutputD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "YoloV3DetectionOutputD") {
            findYoloV3DetectionOutputD = true;
        }
    }
    EXPECT_EQ(findYoloV3DetectionOutputD, true);
}