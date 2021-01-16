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

class split_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "split_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "split_fusion_test TearDown" << std::endl;
  }

  /************************************
   *
   *               split
   *                 /\
   *              /       \
   *         split  ... split
   *           |          |
   *
   *************************************/
  void BuildGraph(ComputeGraphPtr &compute_graph, const vector<int64_t>& shape, int32_t split_dim, int32_t split_num) {
    ge::Graph graph("test_split");
    auto input0 = op::Data("input0");
    ge::Shape shape_x(shape);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT16);
    input0.update_input_desc_x(tensorDescX);
    input0.update_output_desc_y(tensorDescX);

    auto split_dim_op = op::Const("split_dim");
    Tensor axis;
    axis.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));
    axis.SetData((uint8_t*)&split_dim, sizeof(split_dim));
    split_dim_op.set_attr_value(axis);
    split_dim_op.update_output_desc_y(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));

    auto split_layer = op::Split("split");
    split_layer.set_input_split_dim(split_dim_op)
               .set_input_x(input0)
               .set_attr_num_split(split_num)
               .create_dynamic_output_y(split_num);

    std::vector<Operator> inputs{input0, split_dim_op};
    std::vector<Operator> outputs{split_layer};
    graph.SetInputs(inputs).SetOutputs(outputs);
    compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  }
};

TEST_F(split_fusion_test, split_fusion_test_1) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph, {7, 2, 12, 128}, -1, 128);
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ZSplitFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool findSplit = false;
  bool findSplitVD = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "Split") {
        findSplit = true;
        continue;
    }

    if (node->GetType() == "SplitVD") {
      findSplitVD = true;
      continue;
    }
  }
  EXPECT_EQ(findSplit, false);
  EXPECT_EQ(findSplitVD, true);
}

TEST_F(split_fusion_test, split_fusion_test_2) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph, {7, 2, 12, 128}, -1, 2);
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ZSplitFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool findSplit = false;
  bool findSplitVD = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "Split") {
      findSplit = true;
      continue;
    }

    if (node->GetType() == "SplitD") {
      findSplitVD = true;
      continue;
    }
  }
  EXPECT_EQ(findSplit, false);
  EXPECT_EQ(findSplitVD, true);
}

TEST_F(split_fusion_test, split_fusion_test_3) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph, {7, 2, 12, -1}, -1, 128);
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ZSplitFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool findSplit = false;
  bool findSplitVD = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "Split") {
      findSplit = true;
      continue;
    }

    if (node->GetType() == "SplitVD") {
      findSplitVD = true;
      continue;
    }
  }
  
  EXPECT_EQ(findSplit, true);
  EXPECT_EQ(findSplitVD, false);
}

} // namespace fe
