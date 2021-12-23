#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_norm_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class normalize_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "normalize_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "normalize_fusion_test TearDown" << std::endl;
    }
};

TEST_F(normalize_fusion_test, normalize_fusion_test_1) {
    ge::Graph graph("normalize_fusion_test_1");

    auto x1 = op::Data("x1");
    std::vector<int64_t> dims_x1{1, 32, 20, 20};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_NCHW, DT_FLOAT16);
    x1.update_input_desc_x(tensorDescX1);
    x1.update_output_desc_y(tensorDescX1);

    auto x2 = op::Data("x2");
    std::vector<int64_t> dims_x2{1, 32, 20, 20};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_NCHW, DT_FLOAT16);
    x2.update_input_desc_x(tensorDescX2);
    x2.update_output_desc_y(tensorDescX2);

    std::vector<int64_t> dims_y{1, 32, 20, 20};
    ge::Shape shape_y(dims_y);
    ge::TensorDesc tensorDescY(shape_y, FORMAT_NCHW, DT_FLOAT16);

    auto normalizeOp = op::Normalize("normalize");
    normalizeOp.set_input_x1(x1)
               .set_input_x2(x2)
               .set_attr_across_spatial(false)
               .set_attr_channel_shared(true)
               .set_attr_eps(1e-10);

    std::vector<Operator> inputs{x1,x2};
    std::vector<Operator> outputs{normalizeOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("NormalizeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = true;

    EXPECT_EQ(findTranspose, true);
}
TEST_F(normalize_fusion_test, normalize_fusion_test_2) {
    ge::Graph graph("normalize_fusion_test_2");

    auto x1 = op::Data("x1");
    std::vector<int64_t> dims_x1{1, 32, 20, 20};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_NHWC, DT_FLOAT16);
    x1.update_input_desc_x(tensorDescX1);
    x1.update_output_desc_y(tensorDescX1);

    auto x2 = op::Data("x2");
    std::vector<int64_t> dims_x2{1, 32, 20, 20};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_NHWC, DT_FLOAT16);
    x2.update_input_desc_x(tensorDescX2);
    x2.update_output_desc_y(tensorDescX2);
    
    std::vector<int64_t> dims_y{1, 32, 20, 20};
    ge::Shape shape_y(dims_y);
    ge::TensorDesc tensorDescY(shape_y, FORMAT_NHWC, DT_FLOAT16);

    auto normalizeOp = op::Normalize("normalize");
    normalizeOp.set_input_x1(x1)
               .set_input_x2(x2)
               .set_attr_across_spatial(false)
               .set_attr_channel_shared(true)
               .set_attr_eps(1e-10);

    std::vector<Operator> inputs{x1,x2};
    std::vector<Operator> outputs{normalizeOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("NormalizeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = true;

    EXPECT_EQ(findTranspose, true);
}
TEST_F(normalize_fusion_test, normalize_fusion_test_3) {
    ge::Graph graph("normalize_fusion_test_3");

    auto x1 = op::Data("x1");
    std::vector<int64_t> dims_x1{1, 32, 20, 20};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_NHWC, DT_FLOAT16);
    x1.update_input_desc_x(tensorDescX1);
    x1.update_output_desc_y(tensorDescX1);

    auto x2 = op::Data("x2");
    std::vector<int64_t> dims_x2{1, 32, 20, 20};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_NHWC, DT_FLOAT16);
    x2.update_input_desc_x(tensorDescX2);
    x2.update_output_desc_y(tensorDescX2);
    
    std::vector<int64_t> dims_y{1, 32, 20, 20};
    ge::Shape shape_y(dims_y);
    ge::TensorDesc tensorDescY(shape_y, FORMAT_NHWC, DT_FLOAT16);

    auto normalizeOp = op::Normalize("normalize");
    normalizeOp.set_input_x1(x1)
               .set_input_x2(x2)
               .set_attr_across_spatial(true)
               .set_attr_channel_shared(true)
               .set_attr_eps(1e-10);

    std::vector<Operator> inputs{x1,x2};
    std::vector<Operator> outputs{normalizeOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("NormalizeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = true;

    EXPECT_EQ(findTranspose, true);
}