#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class dynamic_lstm_v2_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        
        std::cout << "dynamic_lstm_v2_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dynamic_lstm_v2_fusion_test TearDown" << std::endl;
    }
};

TEST_F(dynamic_lstm_v2_fusion_test, dynamic_lstm_v2_fusion_test_1) {
    ge::Graph graph("dynamic_lstm_v2_fusion_test_1");

    auto xData = op::Data("xData");
    std::vector<int64_t> dims_x{75, 1, 512};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    auto wxData = op::Data("wxData");
    std::vector<int64_t> dims_w{1024, 512};
    ge::Shape shape_w(dims_w);
    ge::TensorDesc tensorDescW(shape_w, FORMAT_NCHW,  DT_FLOAT);
    wxData.update_input_desc_x(tensorDescW);
    wxData.update_output_desc_y(tensorDescW);

    auto whData = op::Data("whData");
    std::vector<int64_t> dims_wh{1024, 256};
    ge::Shape shape_wh(dims_wh);
    ge::TensorDesc tensorDescWh(shape_wh, FORMAT_NCHW,  DT_FLOAT);
    whData.update_input_desc_x(tensorDescWh);
    whData.update_output_desc_y(tensorDescWh);

    auto bData = op::Data("bData");
    std::vector<int64_t> dims_b{1024,};
    ge::Shape shape_b(dims_b);
    ge::TensorDesc tensorDescB(shape_b, FORMAT_NCHW,  DT_FLOAT);
    bData.update_input_desc_x(tensorDescB);
    bData.update_output_desc_y(tensorDescB);

    auto contData = op::Data("contData");
    std::vector<int64_t> dims_cont{75, 1};
    ge::Shape shape_cont(dims_cont);
    ge::TensorDesc tensorDescCont(shape_cont, FORMAT_NCHW,  DT_FLOAT);
    contData.update_input_desc_x(tensorDescCont);
    contData.update_output_desc_y(tensorDescCont);

    auto LSTMOp = op::LSTM("LSTM");
    LSTMOp.set_input_x(xData)
         .set_input_cont(contData)
         .set_input_w_x(wxData)
         .set_input_bias(bData)
         .set_input_w_h(whData)
         .set_attr_num_output(256)
         .set_attr_expose_hidden(false);

    std::vector<Operator> inputs{xData, contData, wxData, bData, whData};
    std::vector<Operator> outputs{LSTMOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("DynamicLSTMFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        std::cout << node->GetType() << std::endl;        
        if (node->GetType() == "DynamicLSTMV2") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, false);
}
