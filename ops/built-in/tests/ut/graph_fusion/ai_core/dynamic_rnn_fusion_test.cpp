#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "rnn.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class dynamic_rnn_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_rnn_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dynamic_rnn_fusion_test TearDown" << std::endl;
    }
};

TEST_F(dynamic_rnn_fusion_test, dynamic_rnn_fusion_test_1) {
    ge::Graph graph("dynamic_rnn_fusion_test_1");

    auto xData = op::Data("xData");
    std::vector<int64_t> dims_x{128, 25, 512};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_FLOAT);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    auto wData = op::Data("wData");
    std::vector<int64_t> dims_w{768, 1024};
    ge::Shape shape_w(dims_w);
    ge::TensorDesc tensorDescW(shape_w, FORMAT_HWCN,  DT_FLOAT);
    wData.update_input_desc_x(tensorDescW);
    wData.update_output_desc_y(tensorDescW);

    auto bData = op::Data("bData");
    std::vector<int64_t> dims_b{1024,};
    ge::Shape shape_b(dims_b);
    ge::TensorDesc tensorDescB(shape_b, FORMAT_NHWC,  DT_FLOAT);
    bData.update_input_desc_x(tensorDescB);
    bData.update_output_desc_y(tensorDescB);

    auto dynamicRNNOp = op::DynamicRNN("DynamicRNN_1");
    dynamicRNNOp.set_input_x(xData)
         .set_input_w(wData)
         .set_input_b(bData)
         .set_attr_time_major(false);

    std::vector<Operator> inputs{xData, wData, bData};
    std::vector<Operator> outputs{dynamicRNNOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("DynamicRNNInsertTransposePass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransposeD") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, true);
}
