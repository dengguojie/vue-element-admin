#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_detect_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "selection_ops.h"
#include "nn_norm_ops.h"

using namespace ge;
using namespace op;

class batch_multi_class_non_max_suppression_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "batch_multi_class_non_max_suppression_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "batch_multi_class_non_max_suppression_fusion_test TearDown" << std::endl;
    }
};

TEST_F(batch_multi_class_non_max_suppression_fusion_test, batch_multi_class_non_max_suppression_fusion_test_1) {
    ge::Graph graph("batch_multi_class_non_max_suppression_fusion_test_1");

    auto boxesData = op::Data("boxesData");
    std::vector<int64_t> dims_x{2, 1024, 1, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND, DT_FLOAT16);
    boxesData.update_input_desc_x(tensorDescX);
    boxesData.update_output_desc_y(tensorDescX);

    auto scoresData = op::Data("scoresData");
    std::vector<int64_t> dims_score{2, 1024, 4};
    ge::Shape shape_score(dims_score);
    ge::TensorDesc tensorDescScore(shape_score, FORMAT_ND, DT_FLOAT16);
    scoresData.update_input_desc_x(tensorDescScore);
    scoresData.update_output_desc_y(tensorDescScore);

    auto nmsOp = op::BatchMultiClassNonMaxSuppression("BatchMultiClassNonMaxSuppression_1");
    nmsOp.set_input_boxes(boxesData);
    nmsOp.set_input_scores(scoresData);
    nmsOp.set_attr_score_threshold(0.6);
    nmsOp.set_attr_iou_threshold(0.6);
    nmsOp.set_attr_max_size_per_class(100);
    nmsOp.set_attr_max_total_size(100);

    std::vector<Operator> inputs{boxesData, scoresData};
    std::vector<Operator> outputs{nmsOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("BatchMultiClassNonMaxSuppressionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, true);
}

TEST_F(batch_multi_class_non_max_suppression_fusion_test, batch_multi_class_non_max_suppression_fusion_test_2) {
    ge::Graph graph("batch_multi_class_non_max_suppression_fusion_test_2");

    auto inputData = op::Data("inputData");
    std::vector<int64_t> dims_score{2, 1024, 4};
    ge::Shape shape_score(dims_score);
    ge::TensorDesc tensorDescScore(shape_score, FORMAT_ND, DT_FLOAT16);
    inputData.update_input_desc_x(tensorDescScore);
    inputData.update_output_desc_y(tensorDescScore);

    ge::Tensor begin_tensor;
    std::vector<int64_t> begin_vec{3};
    ge::Shape begin_shape(begin_vec);
    ge::TensorDesc begin_desc(begin_shape, FORMAT_ND, DT_INT32);
    int32_t begin_size = begin_desc.GetShape().GetShapeSize();
    begin_desc.SetSize(begin_size * sizeof(int32_t));
    begin_tensor.SetTensorDesc(begin_desc);
    int32_t* begin_data = nullptr;
    begin_data = new int32_t[begin_size];
    *(begin_data + 0) = 0;
    *(begin_data + 1) = 0;
    *(begin_data + 2) = 0;
    begin_tensor.SetData((uint8_t*)begin_data, begin_size * sizeof(int32_t));
    delete [] begin_data;

    ge::Tensor end_tensor;
    std::vector<int64_t> end_vec{3};
    ge::Shape end_shape(end_vec);
    ge::TensorDesc end_desc(end_shape, FORMAT_ND, DT_INT32);
    int32_t end_size = end_desc.GetShape().GetShapeSize();
    end_desc.SetSize(end_size * sizeof(int32_t));
    end_tensor.SetTensorDesc(end_desc);
    int32_t* end_data = nullptr;
    end_data = new int32_t[end_size];
    *(end_data + 0) = 2;
    *(end_data + 1) = 1024;
    *(end_data + 2) = 4;
    end_tensor.SetData((uint8_t*)end_data, end_size * sizeof(int32_t));
    delete [] end_data;

    auto begin = op::Const().set_attr_value(begin_tensor);
    auto end = op::Const().set_attr_value(end_tensor);
    auto slice_op = op::Slice("slice_op")
                            .set_input_x(inputData)
                            .set_input_offsets(begin)
                            .set_input_size(end);

    auto boxesData = op::Data("boxesData");
    std::vector<int64_t> dims_x{2, 1024, 1, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND, DT_FLOAT16);
    boxesData.update_input_desc_x(tensorDescX);
    boxesData.update_output_desc_y(tensorDescX);

    auto nmsOp = op::BatchMultiClassNonMaxSuppression("BatchMultiClassNonMaxSuppression_1");
    nmsOp.set_input_boxes(boxesData);
    nmsOp.set_input_scores(slice_op);
    nmsOp.set_attr_score_threshold(0.6);
    nmsOp.set_attr_iou_threshold(0.6);
    nmsOp.set_attr_max_size_per_class(100);
    nmsOp.set_attr_max_total_size(100);

    std::vector<Operator> inputs{boxesData, inputData};
    std::vector<Operator> outputs{nmsOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("BatchMultiClassNonMaxSuppressionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, true);
}

TEST_F(batch_multi_class_non_max_suppression_fusion_test, batch_multi_class_non_max_suppression_fusion_test_3) {
    ge::Graph graph("batch_multi_class_non_max_suppression_fusion_test_3");

    auto inputData = op::Data("inputData");
    std::vector<int64_t> dims_score{2, 1024, 4};
    ge::Shape shape_score(dims_score);
    ge::TensorDesc tensorDescScore(shape_score, FORMAT_ND, DT_FLOAT16);
    inputData.update_input_desc_x(tensorDescScore);
    inputData.update_output_desc_y(tensorDescScore);

    ge::Tensor begin_tensor;
    std::vector<int64_t> begin_vec{3};
    ge::Shape begin_shape(begin_vec);
    ge::TensorDesc begin_desc(begin_shape, FORMAT_ND, DT_FLOAT);
    int32_t begin_size = begin_desc.GetShape().GetShapeSize();
    begin_desc.SetSize(begin_size * sizeof(int32_t));
    begin_tensor.SetTensorDesc(begin_desc);
    int32_t* begin_data = nullptr;
    begin_data = new int32_t[begin_size];
    *(begin_data + 0) = 0.;
    *(begin_data + 1) = 0.;
    *(begin_data + 2) = 0.;
    begin_tensor.SetData((uint8_t*)begin_data, begin_size * sizeof(int32_t));
    delete [] begin_data;

    ge::Tensor end_tensor;
    std::vector<int64_t> end_vec{3};
    ge::Shape end_shape(end_vec);
    ge::TensorDesc end_desc(end_shape, FORMAT_ND, DT_FLOAT);
    int32_t end_size = end_desc.GetShape().GetShapeSize();
    end_desc.SetSize(end_size * sizeof(int32_t));
    end_tensor.SetTensorDesc(end_desc);
    int32_t* end_data = nullptr;
    end_data = new int32_t[end_size];
    *(end_data + 0) = 2.;
    *(end_data + 1) = 1024.;
    *(end_data + 2) = 4.;
    end_tensor.SetData((uint8_t*)end_data, end_size * sizeof(int32_t));
    delete [] end_data;

    auto begin = op::Const().set_attr_value(begin_tensor);
    auto end = op::Const().set_attr_value(end_tensor);
    auto slice_op = op::Slice("slice_op")
                            .set_input_x(inputData)
                            .set_input_offsets(begin)
                            .set_input_size(end);

    auto boxesData = op::Data("boxesData");
    std::vector<int64_t> dims_x{2, 1024, 1, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND, DT_FLOAT16);
    boxesData.update_input_desc_x(tensorDescX);
    boxesData.update_output_desc_y(tensorDescX);

    auto nmsOp = op::BatchMultiClassNonMaxSuppression("BatchMultiClassNonMaxSuppression_1");
    nmsOp.set_input_boxes(boxesData);
    nmsOp.set_input_scores(slice_op);
    nmsOp.set_attr_score_threshold(0.6);
    nmsOp.set_attr_iou_threshold(0.6);
    nmsOp.set_attr_max_size_per_class(100);
    nmsOp.set_attr_max_total_size(100);

    std::vector<Operator> inputs{boxesData, inputData};
    std::vector<Operator> outputs{nmsOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("BatchMultiClassNonMaxSuppressionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, true);
}

TEST_F(batch_multi_class_non_max_suppression_fusion_test, batch_multi_class_non_max_suppression_fusion_test_4) {
    ge::Graph graph("batch_multi_class_non_max_suppression_fusion_test_4");

    auto inputData = op::Data("inputData");
    std::vector<int64_t> dims_score{2, 1024, 4};
    ge::Shape shape_score(dims_score);
    ge::TensorDesc tensorDescScore(shape_score, FORMAT_ND, DT_FLOAT16);
    inputData.update_input_desc_x(tensorDescScore);
    inputData.update_output_desc_y(tensorDescScore);

    auto slice_op = op::SliceD("slice_op")
                            .set_input_x(inputData)
                            .set_attr_offsets({0, 0, 0})
                            .set_attr_size({2, 1024, 4});

    auto boxesData = op::Data("boxesData");
    std::vector<int64_t> dims_x{2, 1024, 1, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND, DT_FLOAT16);
    boxesData.update_input_desc_x(tensorDescX);
    boxesData.update_output_desc_y(tensorDescX);

    auto nmsOp = op::BatchMultiClassNonMaxSuppression("BatchMultiClassNonMaxSuppression_1");
    nmsOp.set_input_boxes(boxesData);
    nmsOp.set_input_scores(slice_op);
    nmsOp.set_attr_score_threshold(0.6);
    nmsOp.set_attr_iou_threshold(0.6);
    nmsOp.set_attr_max_size_per_class(100);
    nmsOp.set_attr_max_total_size(100);

    std::vector<Operator> inputs{boxesData, inputData};
    std::vector<Operator> outputs{nmsOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("BatchMultiClassNonMaxSuppressionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, true);
}
