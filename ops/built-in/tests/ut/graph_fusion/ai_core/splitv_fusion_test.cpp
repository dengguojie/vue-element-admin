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

class splitv_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "splitv_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "splitv_fusion_test TearDown" << std::endl;
  }
  void BuildGraph(ComputeGraphPtr &compute_graph, const vector<int64_t>& shape, int32_t splitv_dim, int32_t splitv_num,vector<int32_t> size_splits) {
    ge::Graph graph("test_splitv");
    auto input0 = op::Data("input0");
    ge::Shape shape_x(shape);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT16);
    input0.update_input_desc_x(tensorDescX);
    input0.update_output_desc_y(tensorDescX);
    auto splitv_dim_op = op::Const("splitv_dim");
    Tensor axis;
    axis.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));
    axis.SetData((uint8_t*)&splitv_dim, sizeof(splitv_dim));
    splitv_dim_op.set_attr_value(axis);
    splitv_dim_op.update_output_desc_y(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));
    
    auto size_splits_op = op::Const("size_splits");
    Tensor axis2;
    axis2.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));
    axis2.SetData((uint8_t*)&size_splits, sizeof(size_splits));
    size_splits_op.set_attr_value(axis2);
    size_splits_op.update_output_desc_y(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));

    auto splitv_layer = op::SplitV("splitV");
    splitv_layer.set_input_x(input0)
               .set_input_size_splits(size_splits_op)
               .create_dynamic_output_y(splitv_num)
               .set_attr_num_split(splitv_num)
               .set_input_split_dim(splitv_dim_op);
    std::vector<Operator> inputs{input0,size_splits_op,splitv_dim_op};
    std::vector<Operator> outputs{splitv_layer};
    graph.SetInputs(inputs).SetOutputs(outputs);
    compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  }
};

TEST_F(splitv_fusion_test, splitv_fusion_test_1) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph, {7, 2, 12, 128}, -1, 3,{60,36,32});
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ZSplitVFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool findSplitVD = false;
  bool findSplitV = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "SplitVD") {
        findSplitVD = true;
        continue;
    }

    if (node->GetType() == "SplitV") {
      findSplitV = true;
      continue;
    }
  }
  EXPECT_EQ(findSplitV, false);
  EXPECT_EQ(findSplitVD, true);
}
TEST_F(splitv_fusion_test, splitv_fusion_test_2) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph, {7, 2, 12, 130}, -1, 65,{2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ZSplitVFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool findSplitVD = false;
  bool findSplitV = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "SplitVD") {
        findSplitVD = true;
        continue;
    }

    if (node->GetType() == "SplitV") {
      findSplitV = true;
      continue;
    }
  }
  EXPECT_EQ(findSplitV, false);
  EXPECT_EQ(findSplitVD, true);
}
TEST_F(splitv_fusion_test, splitv_fusion_test_3) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph, {7, 2, 12, 126}, -1, 126,{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1});
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ZSplitVFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool findSplitVD = false;
  bool findSplitV = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "SplitVD") {
        findSplitVD = true;
        continue;
    }

    if (node->GetType() == "SplitV") {
      findSplitV = true;
      continue;
    }
  }
  EXPECT_EQ(findSplitV, false);
  EXPECT_EQ(findSplitVD, true);
}

} // namespace fe