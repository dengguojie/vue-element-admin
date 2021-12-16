#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "selection_ops.h"

using namespace ge;
using namespace op;

class proposal_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "proposal_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "proposal_fusion_test TearDown" << std::endl;
    }
};

TEST_F(proposal_fusion_test, proposal_fusion_test_1) {
    ge::Graph graph("proposal_fusion_test_1");
    auto cls_prob = op::Data("cls_prob");
    auto bbox_delta = op::Data("bbox_delta");
    auto im_info = op::Data("im_info");
    std::vector<int64_t> dims{10, 36};
    ge::Shape shape(dims);
    std::vector<int64_t> dims1{10, 36};
    ge::Shape shape1(dims1);
    std::vector<int64_t> dims2{10, 36};
    ge::Shape shape2(dims2);
    ge::TensorDesc tensorDesc0(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT);
    ge::TensorDesc tensorDesc2(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
    cls_prob.update_input_desc_x(tensorDesc0);
    cls_prob.update_output_desc_y(tensorDesc0);
    bbox_delta.update_input_desc_x(tensorDesc1);
    bbox_delta.update_output_desc_y(tensorDesc1);
    im_info.update_input_desc_x(tensorDesc2);
    im_info.update_output_desc_y(tensorDesc2);
    auto proposal_op = op::Proposal("Proposal_1");
    proposal_op.set_input_cls_prob(cls_prob);
    proposal_op.set_input_bbox_delta(bbox_delta);
    proposal_op.set_input_im_info(im_info);
    std::vector<Operator> inputs{cls_prob, bbox_delta, im_info};
    std::vector<Operator> outputs{proposal_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ProposalFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findproposalD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ProposalD") {
            findproposalD = true;
        }
    }
    EXPECT_EQ(findproposalD, true);
}