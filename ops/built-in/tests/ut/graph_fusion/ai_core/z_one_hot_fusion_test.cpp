#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "selection_ops.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class zonehot_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "zonehot_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "zonehot_fusion_pass_test TearDown" << std::endl;
    }
};

TEST_F(zonehot_fusion_pass_test, zonehot_fusion_pass_test_001) {
    ge::Graph graph("zonehot_fusion_pass_test_001");

    auto x = op::Data("x");
    std::vector<int64_t> dims_x{2048};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_INT32);
    x.update_input_desc_x(tensorDescX);

    auto on_value = op::Data("on_value");
    std::vector<int64_t> dims_on_value{1};
    ge::Shape shape_on_value(dims_on_value);
    ge::TensorDesc tensorDesc_on_value(shape_on_value, FORMAT_ND,  DT_INT32);
    on_value.update_input_desc_x(tensorDesc_on_value);
    
    auto off_value = op::Data("off_value");
    std::vector<int64_t> dims_off_value{1};
    ge::Shape shape_off_value(dims_off_value);
    ge::TensorDesc tensorDesc_off_value(shape_off_value, FORMAT_ND,  DT_INT32);
    off_value.update_input_desc_x(tensorDesc_off_value);

    int32_t depth=2;

    auto y = op::Data("y");
    std::vector<int64_t> dims_y{2,2048};
    ge::Shape shape_b(dims_y);
    ge::TensorDesc tensorDescB(shape_b, FORMAT_NHWC,  DT_INT32);
    y.update_input_desc_x(tensorDescB);
    y.update_output_desc_y(tensorDescB);

    auto OneHotDOp= op::OneHotD("OneHotD_1");
    OneHotDOp.set_input_x(x)
             .set_input_on_value(on_value)
             .set_input_off_value(off_value)
             .set_attr_depth(depth)
             .set_attr_axis(0);

    std::vector<Operator> inputs{x, on_value,off_value};
    std::vector<Operator> outputs{OneHotDOp};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZOneHotFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr,true);

    bool findonthotD = false;
    bool findonehot  = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "OneHotD") {
            findonthotD = true;
        }
        if (node->GetType() == "OneHot") {
            findonehot = true;
        }
    }
    EXPECT_EQ(findonthotD, false);
    EXPECT_EQ(findonehot, true);
}
TEST_F(zonehot_fusion_pass_test, zonehot_fusion_pass_test_002) {
    ge::Graph graph("zonehot_fusion_pass_test_002");

    auto x = op::Data("x");
    std::vector<int64_t> dims_x{2048};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_INT32);
    x.update_input_desc_x(tensorDescX);

    auto on_value = op::Data("on_value");
    std::vector<int64_t> dims_on_value{1};
    ge::Shape shape_on_value(dims_on_value);
    ge::TensorDesc tensorDesc_on_value(shape_on_value, FORMAT_ND,  DT_INT32);
    on_value.update_input_desc_x(tensorDesc_on_value);
    
    auto off_value = op::Data("off_value");
    std::vector<int64_t> dims_off_value{1};
    ge::Shape shape_off_value(dims_off_value);
    ge::TensorDesc tensorDesc_off_value(shape_off_value, FORMAT_ND,  DT_INT32);
    off_value.update_input_desc_x(tensorDesc_off_value);

    int32_t depth=2;

    auto y = op::Data("y");
    std::vector<int64_t> dims_y{2048,2};
    ge::Shape shape_b(dims_y);
    ge::TensorDesc tensorDescB(shape_b, FORMAT_NHWC,  DT_INT32);
    y.update_input_desc_x(tensorDescB);
    y.update_output_desc_y(tensorDescB);

    auto OneHotDOp= op::OneHotD("OneHotD_1");
    OneHotDOp.set_input_x(x)
             .set_input_on_value(on_value)
             .set_input_off_value(off_value)
             .set_attr_depth(depth)
             .set_attr_axis(-1);

    std::vector<Operator> inputs{x, on_value,off_value};
    std::vector<Operator> outputs{OneHotDOp};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZOneHotFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr,false);

    bool findonthotD = false;
    bool findonehot  = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "OneHotD") {
            findonthotD = true;
        }
        if (node->GetType() == "OneHot") {
            findonehot = true;
        }
    }
    EXPECT_EQ(findonthotD, true);
    EXPECT_EQ(findonehot, false);
}

