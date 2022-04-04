#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "nn_norm_ops.h"

using namespace ge;
using namespace op;

class softmax_cross_entropy_loss_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "softmax_cross_entropy_loss_fusion SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "softmax_cross_entropy_loss_fusion TearDown" << std::endl;
    }
};

TEST_F(softmax_cross_entropy_loss_fusion_test, softmax_cross_entropy_loss_fusion_test_1) {
    ge::Graph graph("softmax_cross_entropy_loss_fusion_test_1");

    auto scores = op::Data("scores");
    auto labels = op::Data("labels");
    auto weights = op::Data("weights");

    std::vector<int64_t> dims_scores{5, 5, 6};
    std::vector<int64_t> dims_labels{5, 6};
    std::vector<int64_t> dims_weights{5};
    ge::Shape shape_scores(dims_scores);
    ge::Shape shape_labels(dims_labels);
    ge::Shape shape_weights(dims_weights);
    ge::TensorDesc tensorDescScores(shape_scores, FORMAT_ND,  DT_FLOAT);
    ge::TensorDesc tensorDescLabels(shape_scores, FORMAT_ND,  DT_INT32);
    ge::TensorDesc tensorDescWeights(shape_weights, FORMAT_ND,  DT_FLOAT);
    
    scores.update_input_desc_x(tensorDescScores);
    scores.update_output_desc_y(tensorDescScores);
    labels.update_input_desc_x(tensorDescLabels);
    labels.update_output_desc_y(tensorDescLabels);
    weights.update_input_desc_x(tensorDescWeights);
    weights.update_output_desc_y(tensorDescWeights);

    auto softmaxcrossentropyloss= op::SoftmaxCrossEntropyLoss("SoftmaxCrossEntropyLoss_1");
    softmaxcrossentropyloss.set_input_scores(scores)
                           .set_input_labels(labels)
                           .set_input_weights(weights);

    std::vector<Operator> inputs{scores,labels,weights};
    std::vector<Operator> outputs{softmaxcrossentropyloss};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("SoftmaxCrossEntropyLossFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findsoftmaxcrossentropyloss = false;
    bool findsgatherelements = false;
    bool findreducemean = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
      if (node->GetType() == "SoftmaxCrossEntropyLoss") {
        findsoftmaxcrossentropyloss = true;
      } else if (node->GetType() == "GatherElements") {
        findsgatherelements = true;
      } else if (node->GetType() == "ReduceMeanD") {
        findreducemean = true;
      }
    }
    EXPECT_EQ(findsoftmaxcrossentropyloss, true);
    EXPECT_EQ(findsgatherelements, true);
    EXPECT_EQ(findreducemean, true);
}

TEST_F(softmax_cross_entropy_loss_fusion_test, softmax_cross_entropy_loss_fusion_test_2) {
    ge::Graph graph("softmax_cross_entropy_loss_fusion_test_1");

    auto scores = op::Data("scores");
    auto labels = op::Data("labels");
    auto weights = op::Data("weights");

    std::vector<int64_t> dims_scores{5, 5, 6};
    std::vector<int64_t> dims_labels{5, 6};
    std::vector<int64_t> dims_weights{5};
    ge::Shape shape_scores(dims_scores);
    ge::Shape shape_labels(dims_labels);
    ge::Shape shape_weights(dims_weights);
    ge::TensorDesc tensorDescScores(shape_scores, FORMAT_ND,  DT_FLOAT);
    ge::TensorDesc tensorDescLabels(shape_scores, FORMAT_ND,  DT_INT32);
    ge::TensorDesc tensorDescWeights(shape_weights, FORMAT_ND,  DT_FLOAT);
    
    scores.update_input_desc_x(tensorDescScores);
    scores.update_output_desc_y(tensorDescScores);
    labels.update_input_desc_x(tensorDescLabels);
    labels.update_output_desc_y(tensorDescLabels);
    weights.update_input_desc_x(tensorDescWeights);
    weights.update_output_desc_y(tensorDescWeights);
    
    string reduction = "sum";
    
    auto softmaxcrossentropyloss= op::SoftmaxCrossEntropyLoss("SoftmaxCrossEntropyLoss_1");
    softmaxcrossentropyloss.set_input_scores(scores)
                           .set_input_labels(labels)
                           .set_input_weights(weights)
                           .set_attr_reduction(reduction);

    std::vector<Operator> inputs{scores,labels,weights};
    std::vector<Operator> outputs{softmaxcrossentropyloss};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("SoftmaxCrossEntropyLossFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findsoftmaxcrossentropyloss = false;
    bool findsgatherelements = false;
    bool findreducesum = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
      if (node->GetType() == "SoftmaxCrossEntropyLoss") {
        findsoftmaxcrossentropyloss = true;
      } else if (node->GetType() == "GatherElements") {
        findsgatherelements = true;
      } else if (node->GetType() == "ReduceSumD") {
        findreducesum = true;
      }
    }
    EXPECT_EQ(findsoftmaxcrossentropyloss, true);
    EXPECT_EQ(findsgatherelements, true);
    EXPECT_EQ(findreducesum, true);
}