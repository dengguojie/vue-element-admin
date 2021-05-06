#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_norm_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class sparse_softmax_cross_entropy_with_logits_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "sparse_softmax_cross_entropy_with_logits_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "sparse_softmax_cross_entropy_with_logits_fusion_test TearDown" << std::endl;
    }
};

TEST_F(sparse_softmax_cross_entropy_with_logits_fusion_test, sparse_softmax_cross_entropy_with_logits_fusion_test_1) {

  // features
  auto data_features = op::Data("features");
  std::vector<int64_t> data_features_vec{2,3};
  ge::Shape data_features_shape(data_features_vec);
  ge::TensorDesc data_features_desc(data_features_shape, FORMAT_ND, DT_FLOAT);
  data_features.update_input_desc_x(data_features_desc);
  data_features.update_output_desc_y(data_features_desc);

  // labels
  auto data_labels = op::Data("labels");
  std::vector<int64_t> data_labels_vec{2,};
  ge::Shape data_labels_shape(data_labels_vec);
  ge::TensorDesc data_labels_desc(data_labels_shape, FORMAT_ND, DT_INT32);
  data_labels.update_input_desc_x(data_labels_desc);
  data_labels.update_output_desc_y(data_labels_desc);

  auto sparse_op = op::SparseSoftmaxCrossEntropyWithLogits("SparseSoftmaxCrossEntropyWithLogits");
  sparse_op.set_input_features(data_features)
           .set_input_labels(data_labels);

  std::vector<Operator> inputs{data_features, data_labels};
  std::vector<Operator> outputs{sparse_op};

  ge::Graph graph("sparse_softmax_cross_entropy_with_logits_fusion_test_1");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SparseSoftMaxFusionPass",
                                               fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOneHotD = false;
  bool findSoftmaxCrossEntropyWithLogits = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SoftmaxCrossEntropyWithLogits") {
            findSoftmaxCrossEntropyWithLogits = true;
        }

        if (node->GetType() == "OneHotD") {
            findOneHotD = true;
        }
    }

  EXPECT_EQ(findOneHotD, true);
  EXPECT_EQ(findSoftmaxCrossEntropyWithLogits, true);
}
TEST_F(sparse_softmax_cross_entropy_with_logits_fusion_test, sparse_softmax_cross_entropy_with_logits_fusion_test_2) {

  // features
  auto data_features = op::Data("features");
  std::vector<int64_t> data_features_vec{-1,-1};
  ge::Shape data_features_shape(data_features_vec);
  ge::TensorDesc data_features_desc(data_features_shape, FORMAT_ND, DT_FLOAT);
  data_features.update_input_desc_x(data_features_desc);
  data_features.update_output_desc_y(data_features_desc);

  // labels
  auto data_labels = op::Data("labels");
  std::vector<int64_t> data_labels_vec{-1,};
  ge::Shape data_labels_shape(data_labels_vec);
  ge::TensorDesc data_labels_desc(data_labels_shape, FORMAT_ND, DT_INT32);
  data_labels.update_input_desc_x(data_labels_desc);
  data_labels.update_output_desc_y(data_labels_desc);

  auto sparse_op = op::SparseSoftmaxCrossEntropyWithLogits("SparseSoftmaxCrossEntropyWithLogits");
  sparse_op.set_input_features(data_features)
           .set_input_labels(data_labels);

  std::vector<Operator> inputs{data_features, data_labels};
  std::vector<Operator> outputs{sparse_op};

  ge::Graph graph("sparse_softmax_cross_entropy_with_logits_fusion_test_1");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SoftmaxFusionPass",
                                               fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOneHotD = false;
  bool findSoftmaxCrossEntropyWithLogits = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SoftmaxCrossEntropyWithLogits") {
            findSoftmaxCrossEntropyWithLogits = true;
        }

        if (node->GetType() == "OneHotD") {
            findOneHotD = true;
        }
    }

  EXPECT_EQ(findOneHotD, false);
  EXPECT_EQ(findSoftmaxCrossEntropyWithLogits, false);
}