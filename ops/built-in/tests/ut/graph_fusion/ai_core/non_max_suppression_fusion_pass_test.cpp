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

class non_max_suppression_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "non_max_suppression_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "non_max_suppression_fusion_pass_test TearDown" << std::endl;
    }
};

TEST_F(non_max_suppression_fusion_pass_test, non_max_suppression_fusion_pass_test_1) {
    ge::Graph graph("non_max_suppression_fusion_pass_test_1");

    auto boxesData = op::Data("boxesData");
    std::vector<int64_t> dims_x{2, 6, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_FLOAT16);
    boxesData.update_input_desc_x(tensorDescX);
    boxesData.update_output_desc_y(tensorDescX);

    auto scoresData = op::Data("scoresData");
    std::vector<int64_t> dims_score{2, 1, 6};
    ge::Shape shape_score(dims_score);
    ge::TensorDesc tensorDescScore(shape_score, FORMAT_ND,  DT_FLOAT16);
    scoresData.update_input_desc_x(tensorDescScore);
    scoresData.update_output_desc_y(tensorDescScore);

  // max_output_size const
    uint64_t max_output_size_value = 5;
    auto shape_data = vector<int64_t>({1});
    TensorDesc desc_max_output_size(ge::Shape(shape_data), FORMAT_ND, DT_INT64);
    Tensor max_output_size_tensor(desc_max_output_size);
    uint64_t *max_output_tensor_value = new uint64_t[1]{max_output_size_value};
    max_output_size_tensor.SetData((uint8_t *) max_output_tensor_value, sizeof(uint64_t));
    auto  max_output_size_const_op = op::Const("max_output_size").set_attr_value(max_output_size_tensor);

    // iou_threshold const
    TensorDesc desc_iou_threshold(ge::Shape(shape_data), FORMAT_ND, DT_FLOAT);
    Tensor iou_threshold_tensor(desc_iou_threshold);
    float *dataValue = new float[1];
    *dataValue = 0.5;
    iou_threshold_tensor.SetData((uint8_t *)dataValue, 4);
    auto  iou_threshold_const_op = op::Const("iou_threshold").set_attr_value(iou_threshold_tensor);


    // score_threshold const
    TensorDesc desc_score_threshold(ge::Shape(shape_data), FORMAT_ND, DT_FLOAT);
    Tensor score_threshold_tensor(desc_score_threshold);
    float *dataValue2 = new float[1];
    *dataValue2 = 0.6;
    score_threshold_tensor.SetData((uint8_t *)dataValue2, 4);
    auto  score_threshold_const_op = op::Const("score_threshold").set_attr_value(score_threshold_tensor);



    // get input data
    auto nmsOp = op::NonMaxSuppressionV6("NonMaxSuppressionV6");
    nmsOp.set_input_boxes(boxesData);
    nmsOp.set_input_scores(scoresData);
    nmsOp.set_input_max_output_size(max_output_size_const_op);
    nmsOp.set_input_iou_threshold(iou_threshold_const_op);
    nmsOp.set_input_score_threshold(score_threshold_const_op);
    nmsOp.set_attr_center_point_box(0);
    nmsOp.set_attr_max_boxes_size(10);


    std::vector<Operator> inputs{boxesData, scoresData,max_output_size_const_op,iou_threshold_const_op,score_threshold_const_op};
    std::vector<Operator> outputs{nmsOp};
    graph.SetInputs(inputs).SetOutputs(outputs);


    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("NonMaxSuppressionV6Fusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);


    bool findNonMaxSuppressionV6 = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "NonMaxSuppressionV7") {
            findNonMaxSuppressionV6 = true;
        }
    }
    EXPECT_EQ(findNonMaxSuppressionV6, true);
    delete[] max_output_tensor_value;
}

TEST_F(non_max_suppression_fusion_pass_test, non_max_suppression_fusion_pass_test_2) {
    ge::Graph graph("non_max_suppression_fusion_pass_test_1");

    auto boxesData = op::Data("boxesData");
    std::vector<int64_t> dims_x{2, 6, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_FLOAT16);
    boxesData.update_input_desc_x(tensorDescX);
    boxesData.update_output_desc_y(tensorDescX);

    auto scoresData = op::Data("scoresData");
    std::vector<int64_t> dims_score{2, 1, 6};
    ge::Shape shape_score(dims_score);
    ge::TensorDesc tensorDescScore(shape_score, FORMAT_ND,  DT_FLOAT16);
    scoresData.update_input_desc_x(tensorDescScore);
    scoresData.update_output_desc_y(tensorDescScore);

  // max_output_size const
    uint32_t max_output_size_value = 5;
    auto shape_data = vector<int64_t>({1});
    TensorDesc desc_max_output_size(ge::Shape(shape_data), FORMAT_ND, DT_INT32);
    Tensor max_output_size_tensor(desc_max_output_size);
    uint32_t *max_output_tensor_value = new uint32_t[1]{max_output_size_value};
    max_output_size_tensor.SetData((uint8_t *) max_output_tensor_value, sizeof(uint32_t));
    auto  max_output_size_const_op = op::Const("max_output_size").set_attr_value(max_output_size_tensor);

    // iou_threshold const
    TensorDesc desc_iou_threshold(ge::Shape(shape_data), FORMAT_ND, DT_FLOAT);
    Tensor iou_threshold_tensor(desc_iou_threshold);
    float *dataValue = new float[1];
    *dataValue = 0.5;
    iou_threshold_tensor.SetData((uint8_t *)dataValue, 4);
    auto  iou_threshold_const_op = op::Const("iou_threshold").set_attr_value(iou_threshold_tensor);


    // score_threshold const
    TensorDesc desc_score_threshold(ge::Shape(shape_data), FORMAT_ND, DT_FLOAT);
    Tensor score_threshold_tensor(desc_score_threshold);
    float *dataValue2 = new float[1];
    *dataValue2 = 0.6;
    score_threshold_tensor.SetData((uint8_t *)dataValue2, 4);
    auto  score_threshold_const_op = op::Const("score_threshold").set_attr_value(score_threshold_tensor);



    // get input data
    auto nmsOp = op::NonMaxSuppressionV6("NonMaxSuppressionV6");
    nmsOp.set_input_boxes(boxesData);
    nmsOp.set_input_scores(scoresData);
    nmsOp.set_input_max_output_size(max_output_size_const_op);
    nmsOp.set_input_iou_threshold(iou_threshold_const_op);
    nmsOp.set_input_score_threshold(score_threshold_const_op);
    nmsOp.set_attr_center_point_box(0);
    nmsOp.set_attr_max_boxes_size(10);


    std::vector<Operator> inputs{boxesData, scoresData,max_output_size_const_op,iou_threshold_const_op,score_threshold_const_op};
    std::vector<Operator> outputs{nmsOp};
    graph.SetInputs(inputs).SetOutputs(outputs);


    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("NonMaxSuppressionV6Fusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);


    bool findNonMaxSuppressionV6 = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "NonMaxSuppressionV7") {
            findNonMaxSuppressionV6 = true;
        }
    }
    EXPECT_EQ(findNonMaxSuppressionV6, true);
    delete[] max_output_tensor_value;
}