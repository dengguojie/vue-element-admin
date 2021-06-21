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

class transposed_update_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "transposed_update_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "transposed_update_fusion_pass_test TearDown" << std::endl;
    }
};

TEST_F(transposed_update_fusion_pass_test, in_white_list_and_no_update) {
    ge::Graph graph("in_white_list_and_no_update");

    auto x = op::Data("x");
    std::vector<int64_t> dims_x{1024, 1024};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_FLOAT);
    x.update_input_desc_x(tensorDescX);

    std::vector<int64_t> perm;
    perm.push_back(1);
    perm.push_back(0);

    auto y = op::Data("y");
    std::vector<int64_t> dims_y{1024, 1024};
    ge::Shape shape_b(dims_y);
    ge::TensorDesc tensorDescB(shape_b, FORMAT_ND,  DT_FLOAT);
    y.update_output_desc_y(tensorDescB);

    auto transposedOp= op::TransposeD("TransposeD_1");
    transposedOp.set_input_x(x)
                .set_attr_perm(perm);

    std::vector<Operator> inputs{x};
    std::vector<Operator> outputs{transposedOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("TransposedUpdateFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr, true);

    bool findTransposed = false;
    bool findTranspose  = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD") {
            findTransposed = false;
        }
        if (node->GetType() == "Transpose") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, true);
    EXPECT_EQ(findTransposed, false);
}

TEST_F(transposed_update_fusion_pass_test, not_in_white_list_and_no_update) {
    ge::Graph graph("not_in_white_list_and_no_update");

    auto x = op::Data("x");
    std::vector<int64_t> dims_x{99999, 77777};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_FLOAT);
    x.update_input_desc_x(tensorDescX);

    std::vector<int64_t> perm;
    perm.push_back(1);
    perm.push_back(0);

    auto y = op::Data("y");
    std::vector<int64_t> dims_y{77777, 99999};
    ge::Shape shape_b(dims_y);
    ge::TensorDesc tensorDescB(shape_b, FORMAT_ND,  DT_FLOAT);
    y.update_output_desc_y(tensorDescB);

    auto transposedOp= op::TransposeD("TransposeD_2");
    transposedOp.set_input_x(x)
                .set_attr_perm(perm);

    std::vector<Operator> inputs{x};
    std::vector<Operator> outputs{transposedOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("TransposedUpdateFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr, false);

    bool findTransposed = false;
    bool findTranspose  = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD") {
            findTransposed = true;
        }
        if (node->GetType() == "Transpose") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, false);
    EXPECT_EQ(findTransposed, true);
}

