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

class zunsortedsegmentsum_update_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "zunsortedsegmentsum_update_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "zunsortedsegmentsum_update_fusion_pass_test TearDown" << std::endl;
    }
};

TEST_F(zunsortedsegmentsum_update_fusion_pass_test, in_white_list_and_update_ng_00) {
    ge::Graph graph("in_white_list_and_update_ng_00");

    auto x = op::Data("x");
    std::vector<int64_t> dims_x{23, 7789, 3, 3};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_INT32);
    x.update_input_desc_x(tensorDescX);

    auto segment_ids = op::Data("segment_ids");
    std::vector<int64_t> dims_segment_ids{23};
    ge::Shape shape_segment_ids(dims_segment_ids);
    ge::TensorDesc tensorDesc_segment_ids(shape_segment_ids, FORMAT_ND,  DT_INT32);
    segment_ids.update_input_desc_x(tensorDesc_segment_ids);

    int32_t num_segments=1000;

    auto y = op::Data("y");
    std::vector<int64_t> dims_y{1000, 7789, 3, 3};
    ge::Shape shape_b(dims_y);
    ge::TensorDesc tensorDescB(shape_b, FORMAT_NHWC,  DT_INT32);
    y.update_input_desc_x(tensorDescB);
    y.update_output_desc_y(tensorDescB);

    auto UnsortedSegmentSumDOp= op::UnsortedSegmentSumD("UnsortedSegmentSumD_1");
    UnsortedSegmentSumDOp.set_input_x(x)
                         .set_input_segment_ids(segment_ids)
                         .set_attr_num_segments(num_segments);

    std::vector<Operator> inputs{x, segment_ids};
    std::vector<Operator> outputs{UnsortedSegmentSumDOp};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZUnsortedSegmentSumUpdateFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr, false);

    bool findUnsortedSegmentSumD = false;
    bool findUnsortedSegmentSum  = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "UnsortedSegmentSumD") {
            findUnsortedSegmentSumD = true;
        }
        if (node->GetType() == "UnsortedSegmentSum") {
            findUnsortedSegmentSum = true;
        }
    }
    EXPECT_EQ(findUnsortedSegmentSum, false);
    EXPECT_EQ(findUnsortedSegmentSumD, true);
}
TEST_F(zunsortedsegmentsum_update_fusion_pass_test, in_white_list_and_update_ok_00) {
    ge::Graph graph("in_white_list_and_update_ok_00");

    auto x = op::Data("x");
    std::vector<int64_t> dims_x{16384, 64};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_FLOAT);
    x.update_input_desc_x(tensorDescX);

    auto segment_ids = op::Data("segment_ids");
    std::vector<int64_t> dims_segment_ids{16384};
    ge::Shape shape_segment_ids(dims_segment_ids);
    ge::TensorDesc tensorDesc_segment_ids(shape_segment_ids, FORMAT_ND,  DT_INT32);
    segment_ids.update_input_desc_x(tensorDesc_segment_ids);

    int32_t num_segments=33;

    auto y = op::Data("y");
    std::vector<int64_t> dims_y{33, 64};
    ge::Shape shape_b(dims_y);
    ge::TensorDesc tensorDescB(shape_b, FORMAT_NHWC,  DT_FLOAT);
    y.update_input_desc_x(tensorDescB);
    y.update_output_desc_y(tensorDescB);

    auto UnsortedSegmentSumDOp= op::UnsortedSegmentSumD("UnsortedSegmentSumD_1");
    UnsortedSegmentSumDOp.set_input_x(x)
                         .set_input_segment_ids(segment_ids)
                         .set_attr_num_segments(num_segments);

    std::vector<Operator> inputs{x, segment_ids};
    std::vector<Operator> outputs{UnsortedSegmentSumDOp};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZUnsortedSegmentSumUpdateFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findUnsortedSegmentSumD = false;
    bool findUnsortedSegmentSum  = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "UnsortedSegmentSumD") {
            findUnsortedSegmentSumD = true;
        }
        if (node->GetType() == "UnsortedSegmentSum") {
            findUnsortedSegmentSum = true;
        }
    }
    EXPECT_EQ(findUnsortedSegmentSum, true);
    EXPECT_EQ(findUnsortedSegmentSumD, false);
}
