#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class unsortedsegmentsum_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "unsortedsegmentsum_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "unsortedsegmentsum_fusion_pass_test TearDown" << std::endl;
  }
};

  //   UnsortedSegmentSum                -->         Concat & UnsortedSegmentSum & Slice

  //                                          concat_dim   x0    x1  x2  x3  x4  x5  x6  x7
  //                                                   \    |    /   /   /   /   /   /   /
  //                                                    \   |   /   /   /   /   /   /   /
  //                                                     Concat
  //                                                        \
  //  x     segment_ids  num_segments                        \    segment_ids  num_segments
  //   \         |         /                                  \         |         /
  //    \        |        /                                    \        |        /
  //     UnsortedSegmentSum              -->                    UnsortedSegmentSum
  //             |                                                      \
  //             |                                                       \   offsets  size
  //             y                                                        \     |     /
  //                                                                       \    |    /
  //                                                                          Slice
  //                                     -->                                    |
  //                                                                            |
  //                                                                            y

TEST_F(unsortedsegmentsum_fusion_pass_test, insert_concat_ok) {
  ge::Graph graph("insert_concat_ok");
  auto x = op::Data("x");
  std::vector<int64_t> dims_x{300, 1};
  ge::Shape shape_x(dims_x);
  ge::TensorDesc tensor_desc_x(shape_x, FORMAT_NHWC,  DT_FLOAT);
  tensor_desc_x.SetOriginShape(shape_x);
  tensor_desc_x.SetOriginFormat(FORMAT_NHWC);
  x.update_input_desc_x(tensor_desc_x);
  x.update_output_desc_y(tensor_desc_x);

  auto segment_ids = op::Data("segment_ids");
  std::vector<int64_t> dims_segment_ids{300};
  ge::Shape shape_segment_ids(dims_segment_ids);
  ge::TensorDesc tensor_desc_segment_ids(shape_segment_ids, FORMAT_NHWC,  DT_INT32);
  tensor_desc_segment_ids.SetOriginShape(shape_segment_ids);
  tensor_desc_segment_ids.SetOriginFormat(FORMAT_NHWC);
  segment_ids.update_input_desc_x(tensor_desc_segment_ids);
  segment_ids.update_output_desc_y(tensor_desc_segment_ids);

  auto num_segments_shape = ge::Shape(std::vector<int64_t>({}));
  TensorDesc tensor_desc_num_segments(num_segments_shape, FORMAT_ND, DT_INT32);
  Tensor num_segments_tensor(tensor_desc_num_segments);
  int32_t num_segments_value = 1000;
  num_segments_tensor.SetData((uint8_t *)(&num_segments_value), sizeof(int32_t));
  auto num_segments = op::Constant().set_attr_value(num_segments_tensor);

  auto unsorted_segment_sum = op::UnsortedSegmentSum("UnsortedSegmentSum");
  unsorted_segment_sum.set_input_x(x)
                      .set_input_segment_ids(segment_ids)
                      .set_input_num_segments(num_segments);
  std::vector<int64_t> dims_out{1000, 1};
  ge::Shape shape_out(dims_out);
  ge::TensorDesc tensor_desc_out(shape_out, FORMAT_NHWC,  DT_FLOAT);
  tensor_desc_out.SetOriginShape(shape_out);
  tensor_desc_out.SetOriginFormat(FORMAT_NHWC);
  unsorted_segment_sum.update_input_desc_x(tensor_desc_x);
  unsorted_segment_sum.update_input_desc_segment_ids(tensor_desc_segment_ids);
  unsorted_segment_sum.update_input_desc_num_segments(tensor_desc_num_segments);
  unsorted_segment_sum.update_output_desc_y(tensor_desc_out);

  auto identity = op::Identity("output");
  identity.set_input_x(unsorted_segment_sum);
  identity.update_input_desc_x(tensor_desc_out);
  identity.update_output_desc_y(tensor_desc_out);

  std::vector<Operator> inputs{x, segment_ids, num_segments};
  std::vector<Operator> outputs{identity};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  ASSERT_EQ(fe::FusionPassTestUtils::RunGraphFusionPass("UnsortedSegmentSumFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr), SUCCESS);

  bool find_concat  = false;
  bool find_unsorted_segment_sum_pad  = false;
  bool find_slice  = false;
  bool find_unsorted_segment_sum = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetName() == "x") {
      EXPECT_EQ(node->GetOutDataNodesSize(), 8);
      for (const auto &out_node : node->GetOutDataNodes()) {
        EXPECT_EQ(out_node->GetType(), "Concat");
        EXPECT_EQ(out_node->GetName(), "UnsortedSegmentSum/Concat");
      }
    } else if ((node->GetName() == "segment_ids") ||
               ((node->GetType() == "Concat") && (node->GetName() == "UnsortedSegmentSum/Concat"))) {
      if (node->GetType() == "Concat") {
        find_concat = true;
      }
      for (const auto &out_node : node->GetOutDataNodes()) {
        EXPECT_EQ(out_node->GetType(), "UnsortedSegmentSum");
        EXPECT_EQ(out_node->GetName(), "UnsortedSegmentSum/UnsortedSegmentSumPad");
      }
    } else if ((node->GetType() == "UnsortedSegmentSum") && (node->GetName() == "UnsortedSegmentSum/UnsortedSegmentSumPad")) {
      find_unsorted_segment_sum_pad = true;
      for (const auto &out_node : node->GetOutDataNodes()) {
        EXPECT_EQ(out_node->GetType(), "Slice");
        EXPECT_EQ(out_node->GetName(), "UnsortedSegmentSum/Slice");
      }
    } else if ((node->GetType() == "Slice") && (node->GetName() == "UnsortedSegmentSum/Slice")) {
      find_slice = true;
      for (const auto &out_node : node->GetOutDataNodes()) {
        EXPECT_EQ(out_node->GetType(), "Identity");
        EXPECT_EQ(out_node->GetName(), "output");
      }
    } else if (node->GetName() == "UnsortedSegmentSum") {
      find_unsorted_segment_sum = true;
    }
  }
  EXPECT_EQ(find_concat, true);
  EXPECT_EQ(find_unsorted_segment_sum_pad, true);
  EXPECT_EQ(find_slice, true);
  EXPECT_EQ(find_unsorted_segment_sum, false);
}
