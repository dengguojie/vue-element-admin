#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_calculation_ops.h"
#include "split_combination_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "fp16_t.hpp"

using namespace ge;

namespace fe {

class splitd_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "splitd_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "splitd_fusion_test TearDown" << std::endl;
  }
  void BuildGraph(ComputeGraphPtr &compute_graph, const vector<int64_t>& shape, int32_t splitd_dim, int32_t splitd_num) {
    ge::Graph graph("test_splitd");
    auto input0 = op::Data("input0");
    ge::Shape shape_x(shape);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT16);
    input0.update_input_desc_x(tensorDescX);
    input0.update_output_desc_y(tensorDescX);
    auto splitd_layer = op::SplitD("splitD");
    splitd_layer.set_input_x(input0)
               .create_dynamic_output_y(splitd_num)
               .set_attr_num_split(splitd_num)
               .set_attr_split_dim(splitd_dim);
    std::vector<Operator> inputs{input0};
    std::vector<Operator> outputs{splitd_layer};
    graph.SetInputs(inputs).SetOutputs(outputs);
    compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  }
};

TEST_F(splitd_fusion_test, splitd_fusion_test_1) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph, {7, 2, 12, 128}, -1, 128);
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ZSplitDFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool findSplitD = false;
  bool findSplitVD = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "SplitD") {
        findSplitD = true;
        continue;
    }

    if (node->GetType() == "SplitVD") {
      findSplitVD = true;
      continue;
    }
  }
  EXPECT_EQ(findSplitD, false);
  EXPECT_EQ(findSplitVD, true);
}
TEST_F(splitd_fusion_test, splitd_fusion_test_2) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph, {7, 2, 12, 128}, -1, 126);
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ZSplitDFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool findSplitD = false;
  bool findSplitVD = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "SplitD") {
        findSplitD = true;
        continue;
    }

    if (node->GetType() == "SplitVD") {
      findSplitVD = true;
      continue;
    }
  }
  EXPECT_EQ(findSplitD, false);
  EXPECT_EQ(findSplitVD, true);
}
TEST_F(splitd_fusion_test, splitd_fusion_test_3) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph, {7, 2, 12, 128}, -1, 63);
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ZSplitDFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool findSplitD = false;
  bool findSplitVD = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "SplitD") {
        findSplitD = true;
        continue;
    }

    if (node->GetType() == "SplitVD") {
      findSplitVD = true;
      continue;
    }
  }
  EXPECT_EQ(findSplitD, true);
  EXPECT_EQ(findSplitVD, false);
}


} // namespace fe