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

class softmaxv2_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "softmaxv2_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "softmaxv2_fusion_pass_test TearDown" << std::endl;
    }
};

TEST_F(softmaxv2_fusion_pass_test, softmax_v2_fusion_pass_test_01) {
    ge::Graph graph("softmaxv2_fusion_pass_test");

    auto x = op::Data("x");
    std::vector<int64_t> dims_x{1000, 5, 64, 64};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_FLOAT);
    x.update_input_desc_x(tensorDescX);

    std::vector<int64_t> axes;
    axes.push_back(1);

    auto SoftmaxV2= op::SoftmaxV2("SoftmaxV2");
    SoftmaxV2.set_input_x(x)
             .set_attr_axes(axes);

    std::vector<Operator> inputs{x};
    std::vector<Operator> outputs{SoftmaxV2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ASoftmaxFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
}
