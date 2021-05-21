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

class unsortedsegmentsumd_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "unsortedsegmentsumd_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "unsortedsegmentsumd_fusion_pass_test TearDown" << std::endl;
  }
};

//
//                                                   Identity
//                                                       |
//           Identity                                 SliceD
//              |                                        |
//     UnsortedSegmentSumD           ==>        UnsortedSegmentSumd8
//         /          \                             /          \
//        x         segment_ids                 ConcatD      segment_ids
//                                             \\\\////
//                                                 x
//
TEST_F(unsortedsegmentsumd_fusion_pass_test, insert_concat_d_ok) {
  ge::Graph graph("insert_concat_d_ok");

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

  int32_t num_segments = 1000;
  auto unsorted_segment_sum_d = op::UnsortedSegmentSumD("UnsortedSegmentSumD");
  unsorted_segment_sum_d.set_input_x(x)
                        .set_input_segment_ids(segment_ids)
                        .set_attr_num_segments(num_segments);
  std::vector<int64_t> dims_out{1000, 1};
  ge::Shape shape_out(dims_out);
  ge::TensorDesc tensor_desc_out(shape_out, FORMAT_NHWC,  DT_FLOAT);
  tensor_desc_out.SetOriginShape(shape_out);
  tensor_desc_out.SetOriginFormat(FORMAT_NHWC);
  unsorted_segment_sum_d.update_input_desc_x(tensor_desc_x);
  unsorted_segment_sum_d.update_input_desc_segment_ids(tensor_desc_segment_ids);
  unsorted_segment_sum_d.update_output_desc_y(tensor_desc_out);

  auto identity = op::Identity("output");
  identity.set_input_x(unsorted_segment_sum_d);
  identity.update_input_desc_x(tensor_desc_out);
  identity.update_output_desc_y(tensor_desc_out);

  std::vector<Operator> inputs{x, segment_ids};
  std::vector<Operator> outputs{identity};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  ASSERT_EQ(fe::FusionPassTestUtils::RunGraphFusionPass("UnsortedSegmentSumdFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr), SUCCESS);

  bool find_concat_d  = false;
  bool find_unsorted_segment_sum_d_pad  = false;
  bool find_slice_d  = false;
  bool find_unsorted_segment_sum_d = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetName() == "x") {
      EXPECT_EQ(node->GetOutDataNodesSize(), 8);
      for (const auto &out_node : node->GetOutDataNodes()) {
        EXPECT_EQ(out_node->GetType(), "ConcatD");
        EXPECT_EQ(out_node->GetName(), "UnsortedSegmentSumD/ConcatD");
      }
    } else if ((node->GetName() == "segment_ids") ||
               ((node->GetType() == "ConcatD") && (node->GetName() == "UnsortedSegmentSumD/ConcatD"))) {
      if (node->GetType() == "ConcatD") {
        find_concat_d = true;
      }
      for (const auto &out_node : node->GetOutDataNodes()) {
        EXPECT_EQ(out_node->GetType(), "UnsortedSegmentSumD");
        EXPECT_EQ(out_node->GetName(), "UnsortedSegmentSumD/UnsortedSegmentSumDPad");
      }
    } else if ((node->GetType() == "UnsortedSegmentSumD") && (node->GetName() == "UnsortedSegmentSumD/UnsortedSegmentSumDPad")) {
      find_unsorted_segment_sum_d_pad = true;
      for (const auto &out_node : node->GetOutDataNodes()) {
        EXPECT_EQ(out_node->GetType(), "SliceD");
        EXPECT_EQ(out_node->GetName(), "UnsortedSegmentSumD/SliceD");
      }
    } else if ((node->GetType() == "SliceD") && (node->GetName() == "UnsortedSegmentSumD/SliceD")) {
      find_slice_d = true;
      for (const auto &out_node : node->GetOutDataNodes()) {
        EXPECT_EQ(out_node->GetType(), "Identity");
        EXPECT_EQ(out_node->GetName(), "output");
      }
    } else if (node->GetName() == "UnsortedSegmentSumD") {
      find_unsorted_segment_sum_d = true;
    }
  }
  EXPECT_EQ(find_concat_d, true);
  EXPECT_EQ(find_unsorted_segment_sum_d_pad, true);
  EXPECT_EQ(find_slice_d, true);
  EXPECT_EQ(find_unsorted_segment_sum_d, false);
}
