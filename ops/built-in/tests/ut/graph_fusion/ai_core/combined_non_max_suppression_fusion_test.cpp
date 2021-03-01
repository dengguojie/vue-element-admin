#include <type_traits>
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

class combined_non_max_suppression_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "combined_non_max_suppression_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "combined_non_max_suppression_fusion_test TearDown" << std::endl;
    }
};

template <typename T>
static Operator GetConstNode(const std::vector<int64_t>& const_shape,
                             const std::vector<T>& const_value,
                             const std::string& const_name,
                             const ge::Format& const_format) {
  auto const_size = const_value.size();
  constexpr ge::DataType const_dtype = std::is_same<T, float>::value ? ge::DT_FLOAT : ge::DT_INT32;
  TensorDesc const_desc(ge::Shape(const_shape), const_format, const_dtype);
  Tensor const_tensor(const_desc);
  const_tensor.SetData(reinterpret_cast<const uint8_t*>(const_value.data()), const_size * sizeof(T));
  auto const_op = op::Const(const_name.c_str()).set_attr_value(const_tensor);
  return const_op;
}

TEST_F(combined_non_max_suppression_fusion_test, combined_non_max_suppression_fusion_test_1) {
    ge::Graph graph("combined_non_max_suppression_fusion_test_1");

    auto boxesData = op::Data("boxesData");
    std::vector<int64_t> dims_x{2, 1024, 1, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_FLOAT16);
    boxesData.update_input_desc_x(tensorDescX);
    boxesData.update_output_desc_y(tensorDescX);

    auto scoresData = op::Data("scoresData");
    std::vector<int64_t> dims_score{2, 1024, 4};
    ge::Shape shape_score(dims_score);
    ge::TensorDesc tensorDescScore(shape_score, FORMAT_ND,  DT_FLOAT16);
    scoresData.update_input_desc_x(tensorDescScore);
    scoresData.update_output_desc_y(tensorDescScore);

    // get a const op
    std::vector<int64_t> const_shape_dims{};
    auto const_format = FORMAT_ND;
    std::vector<int32_t>  max_output_size_per_class_const_value{100};
    std::vector<int32_t>  max_total_size_const_value{100};
    std::vector<float>  iou_threshold_const_value{0.5};
    std::vector<float> score_threshold_const_value{0.1};

    auto max_output_size_per_class_const_op = GetConstNode(const_shape_dims, max_output_size_per_class_const_value, "max_output_size_per_class_const_node", const_format);
    auto max_total_size_const_op = GetConstNode(const_shape_dims, max_total_size_const_value, "max_total_size_const_node", const_format);
    auto iou_threshold_const_op = GetConstNode(const_shape_dims, iou_threshold_const_value, "iou_threshold_const_node", const_format);
    auto score_threshold_const_op = GetConstNode(const_shape_dims, score_threshold_const_value, "score_threshold_const_node", const_format);

    auto nmsOp = op::CombinedNonMaxSuppression("CombinedNonMaxSuppression_1");
    nmsOp.set_input_boxes(boxesData);
    nmsOp.set_input_scores(scoresData);
    nmsOp.set_input_max_output_size_per_class(max_output_size_per_class_const_op);
    nmsOp.set_input_max_total_size(max_total_size_const_op);
    nmsOp.set_input_iou_threshold(iou_threshold_const_op);
    nmsOp.set_input_score_threshold(score_threshold_const_op);

    std::vector<Operator> inputs{boxesData, scoresData, max_output_size_per_class_const_op, max_total_size_const_op, iou_threshold_const_op, score_threshold_const_op};
    std::vector<Operator> outputs{nmsOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("CombinedNonMaxSuppressionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, false);
    // EXPECT_EQ(findTranspose, true);
}

TEST_F(combined_non_max_suppression_fusion_test, combined_non_max_suppression_fusion_test_2) {
    ge::Graph graph("combined_non_max_suppression_fusion_test_2");

    auto boxesData = op::Data("boxesData");
    std::vector<int64_t> dims_x{2, 1024, 4000, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_FLOAT16);
    boxesData.update_input_desc_x(tensorDescX);
    boxesData.update_output_desc_y(tensorDescX);

    auto scoresData = op::Data("scoresData");
    std::vector<int64_t> dims_score{2, 1024, 4000};
    ge::Shape shape_score(dims_score);
    ge::TensorDesc tensorDescScore(shape_score, FORMAT_ND,  DT_FLOAT16);
    scoresData.update_input_desc_x(tensorDescScore);
    scoresData.update_output_desc_y(tensorDescScore);

    // get a const op
    std::vector<int64_t> const_shape_dims{};
    auto const_format = FORMAT_ND;
    std::vector<int32_t>  max_output_size_per_class_const_value{1000};
    std::vector<int32_t>  max_total_size_const_value{1000};
    std::vector<float>  iou_threshold_const_value{0.5};
    std::vector<float> score_threshold_const_value{0.1};

    auto max_output_size_per_class_const_op = GetConstNode(const_shape_dims, max_output_size_per_class_const_value, "max_output_size_per_class_const_node", const_format);
    auto max_total_size_const_op = GetConstNode(const_shape_dims, max_total_size_const_value, "max_total_size_const_node", const_format);
    auto iou_threshold_const_op = GetConstNode(const_shape_dims, iou_threshold_const_value, "iou_threshold_const_node", const_format);
    auto score_threshold_const_op = GetConstNode(const_shape_dims, score_threshold_const_value, "score_threshold_const_node", const_format);

    auto nmsOp = op::CombinedNonMaxSuppression("CombinedNonMaxSuppression_1");
    nmsOp.set_input_boxes(boxesData);
    nmsOp.set_input_scores(scoresData);
    nmsOp.set_input_max_output_size_per_class(max_output_size_per_class_const_op);
    nmsOp.set_input_max_total_size(max_total_size_const_op);
    nmsOp.set_input_iou_threshold(iou_threshold_const_op);
    nmsOp.set_input_score_threshold(score_threshold_const_op);

    std::vector<Operator> inputs{boxesData, scoresData, max_output_size_per_class_const_op, max_total_size_const_op, iou_threshold_const_op, score_threshold_const_op};
    std::vector<Operator> outputs{nmsOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("CombinedNonMaxSuppressionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, false);
}
