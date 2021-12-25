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
/*
1:
SplitV ---> SplitVD

2:
          input                                                             input
            |                                                                 |
       S p l i t V D            ---  -fusion-  --->                    S  p  l  i  t  V  D
      / ...   |        \                                             /         |        \
     /  ...   |         \                                           /          |         \
output_1 ... output_m .. output_n                             SplitVD_1 ... SplitVD_M ... SplitVD_N
                                                             /    |   \    /    |     \    /    |    \
                                                      output_1  output_2 ... output_m  ...           output_n
*/

  void BuildGraph(ComputeGraphPtr &compute_graph, const vector<int64_t>& shape, int32_t split_dim, int32_t split_num) {
    ge::Graph graph("test_splitv");
    // make input x
    auto input0 = op::Data("input0");
    ge::Shape shape_x(shape);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT16);
    input0.update_input_desc_x(tensorDescX);
    input0.update_output_desc_y(tensorDescX);
    // make input split_dim
    auto split_dim_op = op::Const("split_dim");
    Tensor axis;
    axis.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_ND, DT_INT32));
    axis.SetData((uint8_t*)&split_dim, sizeof(split_dim));
    split_dim_op.set_attr_value(axis);
    split_dim_op.update_output_desc_y(TensorDesc(ge::Shape({1}), FORMAT_ND, DT_INT32));
    // make input size_splits
    auto size_splits_op = op::Const("size_splits");
    Tensor axis2;
    int32_t size_value = 1;
    int32_t* size_splits_data = new int32_t[split_num];
    for (size_t i = 0; i < split_num; i++) {
        *(size_splits_data + i) = size_value;
    }
    axis2.SetTensorDesc(TensorDesc(ge::Shape({split_num}), FORMAT_ND, DT_INT32));
    axis2.SetData((uint8_t*)size_splits_data, split_num*sizeof(int32_t));
    size_splits_op.set_attr_value(axis2);
    size_splits_op.update_output_desc_y(TensorDesc(ge::Shape({split_num}), FORMAT_ND, DT_INT32));
    // make splitV input connected
    auto splitv_layer = op::SplitV("splitV");
    splitv_layer.set_input_x(input0)
                .set_input_size_splits(size_splits_op)
                .set_input_split_dim(split_dim_op)
                .create_dynamic_output_y(split_num)
                .set_attr_num_split(split_num);
    std::vector<Operator> inputs{input0, size_splits_op, split_dim_op};
    std::vector<Operator> outputs{splitv_layer};
    graph.SetInputs(inputs).SetOutputs(outputs);
    compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  }
};

TEST_F(splitv_fusion_test, splitv_fusion_test_1) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph, {7, 2, 12, 128}, 3, 128);
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ZSplitVFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool findSplitV = false;
  bool findSplitVD = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "SplitV") {
        findSplitV = true;
        continue;
    }
    if (node->GetType() == "SplitVD") {
      findSplitVD = true;
      continue;
    }
  }
  EXPECT_EQ(findSplitV, false);
  EXPECT_EQ(findSplitVD, true);
}

} // namespace fe
