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

class yolo_v5_detection_output_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "yolo_v5_detection_output_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "yolo_v5_detection_output_fusion_pass_test TearDown" << std::endl;
    }
};

TEST_F(yolo_v5_detection_output_fusion_pass_test, yolo_v5_detection_output_fusion_pass_test_1) {
    ge::Graph graph("yolo_v5_detection_output_fusion_pass_test_1");

    ge::op::Data input[3];
    ge::op::Data img_info;
    ge::op::YoloPreDetection yolo_pre_op[3];
    int64_t fh = 80;
    for (uint32_t i = 0; i < 3; i++) {
        std::string tmp_name = "x_" + std::to_string(i);
        ge::TensorDesc tensorDesc(ge::Shape({1, 256, fh, fh}), ge::FORMAT_ND, ge::DT_FLOAT);
        input[i] = op::Data(tmp_name);
        input[i].update_input_desc_x(tensorDesc);
        input[i].update_output_desc_y(tensorDesc);

        tmp_name = "yolo_pre_detection_" + std::to_string(i);
        yolo_pre_op[i] = op::YoloPreDetection(tmp_name);
        yolo_pre_op[i].set_input_x(input[i]);
        fh /= 2;
    }
    ge::TensorDesc tensorDesc(ge::Shape({1, 4}), ge::FORMAT_ND, ge::DT_FLOAT);
    img_info = op::Data("img_info");
    img_info.update_input_desc_x(tensorDesc);
    img_info.update_output_desc_y(tensorDesc);

    auto yolov5_detout_op = op::YoloV5DetectionOutput("yolo_v5_detection_output_0");
    yolov5_detout_op.create_dynamic_input_x(10);
    for (uint32_t i = 0; i < 3; i++) {
        yolov5_detout_op.set_dynamic_input_x(3 * 0 + i, yolo_pre_op[i], "coord_data");
    }
    for (uint32_t i = 0; i < 3; i++) {
        yolov5_detout_op.set_dynamic_input_x(3 * 1 + i, yolo_pre_op[i], "obj_prob");
    }
    for (uint32_t i = 0; i < 3; i++) {
        yolov5_detout_op.set_dynamic_input_x(3 * 2 + i, yolo_pre_op[i], "classes_prob");
    }
    yolov5_detout_op.set_dynamic_input_x(9, img_info);
    yolov5_detout_op.set_attr_biases({10., 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326});

    std::vector<Operator> inputs;
    std::vector<Operator> outputs{yolov5_detout_op};
    for (uint32_t i = 0; i < 3; i++) {
        inputs.push_back(input[i]);
    }
    inputs.push_back(img_info);

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "YoloV5DetectionOutputPass_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("YoloV5DetectionOutputPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "YoloV5DetectionOutputPass_after");

    bool find_fused_op = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "YoloV5DetectionOutputD") {
            find_fused_op = true;
        }
    }
    EXPECT_EQ(find_fused_op, true);
}
