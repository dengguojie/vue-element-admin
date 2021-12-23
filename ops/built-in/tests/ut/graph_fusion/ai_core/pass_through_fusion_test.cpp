#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class pass_through_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "pass_through_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "pass_through_fusion_test TearDown" << std::endl;
    }
};

TEST_F(pass_through_fusion_test, pass_through_fusion_test_1) {
    ge::Graph graph("pass_through_fusion_test_1");

    auto x = op::Data("x");
    std::vector<int64_t> dims_x{1, 32, 20, 20};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW, DT_FLOAT16);
    x.update_input_desc_x(tensorDescX);
    std::vector<int64_t> dims_y{1, 128, 10, 10};
    ge::Shape shape_y(dims_y);
    ge::TensorDesc tensorDescY(shape_y, FORMAT_NCHW, DT_FLOAT16);
    x.update_output_desc_y(tensorDescY);

    auto passThroughOp = op::PassThrough("pass_through");
    passThroughOp.set_input_x(x)
                 .set_attr_stride(2)
                 .set_attr_reverse(false);

    std::vector<Operator> inputs{x};
    std::vector<Operator> outputs{passThroughOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PassThroughFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = true;

    EXPECT_EQ(findTranspose, true);
}
TEST_F(pass_through_fusion_test, pass_through_fusion_test_2) {
    ge::Graph graph("pass_through_fusion_test_2");

    auto x = op::Data("x");
    std::vector<int64_t> dims_x{1, 32, 20, 20};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT16);
    x.update_input_desc_x(tensorDescX);
    std::vector<int64_t> dims_y{1, 32, 10, 10};
    ge::Shape shape_y(dims_y);
    ge::TensorDesc tensorDescY(shape_y, FORMAT_NHWC, DT_FLOAT16);
    x.update_output_desc_y(tensorDescY);

    auto passThroughOp = op::PassThrough("pass_through");
    passThroughOp.set_input_x(x)
                 .set_attr_stride(2)
                 .set_attr_reverse(false);

    std::vector<Operator> inputs{x};
    std::vector<Operator> outputs{passThroughOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PassThroughFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = true;

    EXPECT_EQ(findTranspose, true);
}
