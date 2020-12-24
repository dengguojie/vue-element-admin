#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class unpack_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "inplace_add TearDown" << std::endl;
    }
};

TEST_F(unpack_fusion_test, unpack_fusion_test_001) {
    ge::Graph graph("unpack_fusion_test_001");
    auto unpack_input_data = op::Data("unpack_input_data");
    std::vector<int64_t> dims{16, 2, 16, 16};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    unpack_input_data.update_input_desc_x(tensorDesc);
    unpack_input_data.update_output_desc_y(tensorDesc);
    auto unpack_op = op::Unpack("unpack_0");
    unpack_op.set_input_x(unpack_input_data);
    unpack_op.set_attr_num(2);
    unpack_op.set_attr_axis(1);

    std::vector<Operator> inputs{unpack_input_data};
    std::vector<Operator> outputs{unpack_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("UnpackFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool unpack_match = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Unpack") {
            unpack_match = true;
        }
    }
    EXPECT_EQ(unpack_match, true);
}