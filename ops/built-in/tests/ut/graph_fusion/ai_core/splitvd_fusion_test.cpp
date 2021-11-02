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

class splitvd_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "splitvd_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "splitvd_fusion_test TearDown" << std::endl;
  }
  void BuildGraph(ComputeGraphPtr &compute_graph, const vector<int64_t>& shape, int32_t splitvd_dim, int32_t splitvd_num,vector<int64_t> size_splits) {
    ge::Graph graph("test_splitvd");
    auto input0 = op::Data("input0");
    ge::Shape shape_x(shape);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT16);
    input0.update_input_desc_x(tensorDescX);
    input0.update_output_desc_y(tensorDescX);

    auto splitvd_layer = op::SplitVD("splitVD");
    splitvd_layer.set_input_x(input0)
               .create_dynamic_output_y(splitvd_num)
               .set_attr_size_splits(size_splits)
               .set_attr_num_split(splitvd_num)
               .set_attr_split_dim(splitvd_dim);
    std::vector<Operator> inputs{input0};
    std::vector<Operator> outputs{splitvd_layer};
    graph.SetInputs(inputs).SetOutputs(outputs);
    compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  }
};

TEST_F(splitvd_fusion_test, splitvd_fusion_test_1) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph, {7, 2, 12, 128}, -1, 3,{60,36,32});
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ZSplitVDFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool findSplitVD = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "SplitVD") {
        findSplitVD = true;
        continue;
    }
  }
  EXPECT_EQ(findSplitVD, true);
}
TEST_F(splitvd_fusion_test, splitvd_fusion_test_2) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph, {7, 2, 12, 130}, -1, 65,{2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ZSplitVDFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool findSplitVD = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "SplitVD") {
        findSplitVD = true;
        continue;
    }
  }
  EXPECT_EQ(findSplitVD, true);
}
TEST_F(splitvd_fusion_test, splitvd_fusion_test_3) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph, {7, 2, 12, 126}, -1, 126,{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1});
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ZSplitVDFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool findSplitVD = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "SplitVD") {
        findSplitVD = true;
        continue;
    }
  }
  EXPECT_EQ(findSplitVD, true);
}

} // namespace fe