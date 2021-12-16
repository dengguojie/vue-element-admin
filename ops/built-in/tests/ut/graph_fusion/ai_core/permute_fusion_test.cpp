#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "transformation_ops.h"

using namespace ge;
using namespace op;

class permute_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "permute_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "permute_fusion_test TearDown" << std::endl;
    }
};

TEST_F(permute_fusion_test, permute_fusion_test_1) {
    ge::Graph graph("permute_fusion_test_1");
    auto xdata = op::Data("x");
    std::vector<int64_t> dims{10, 20, 30, 40};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc0(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    xdata.update_input_desc_x(tensorDesc0);
    xdata.update_output_desc_y(tensorDesc0);
    auto permute_op = op::Permute("Permute_1");
    permute_op.set_input_x(xdata)
                .set_attr_order({0,3,2,1});
    std::vector<Operator> inputs{xdata};
    std::vector<Operator> outputs{permute_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PermuteFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findPermute = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD") {
            findPermute = true;
        }
    }
    EXPECT_EQ(findPermute, true);
}