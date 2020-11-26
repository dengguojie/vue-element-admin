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
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_FLOAT16);
    boxesData.update_input_desc_x(tensorDescX);
    boxesData.update_output_desc_y(tensorDescX);

    auto scoresData = op::Data("scoresData");
    std::vector<int64_t> dims_score{2, 1024, 4};
    ge::Shape shape_score(dims_score);
    ge::TensorDesc tensorDescScore(shape_score, FORMAT_ND,  DT_FLOAT16);
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
