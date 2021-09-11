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
namespace fe{
class lstm_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
    std::cout << "lstm_fusion_test SetUp" << std::endl;
}

    static void TearDownTestCase() {
    std::cout << "lstm_fusion_test TearDown" << std::endl;
}
};

TEST_F(lstm_fusion_test, lstm_fusion_test_1) {
    ge::Graph graph("lstm_fusion_test_1");

    auto xData = op::Data("x");
    std::vector<int64_t> dims_x{20, 1, 96};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    auto wxData = op::Const("w_x");
    std::vector<int64_t> dims_w{288, 72};
    ge::Shape shape_w(dims_w);
    ge::TensorDesc tensorDescW(shape_w, FORMAT_NCHW,  DT_FLOAT);
    Tensor wx;
    float* wx_value = new float[288*72];
    wx.SetTensorDesc(tensorDescW);
    wx.SetData((uint8_t*)wx_value, 288*72*sizeof(float));
    wxData.set_attr_value(wx);
    wxData.update_output_desc_y(tensorDescW);

    auto whData = op::Const("w_h");
    std::vector<int64_t> dims_wh{288, 96};
    ge::Shape shape_wh(dims_wh);
    ge::TensorDesc tensorDescWh(shape_wh, FORMAT_NCHW, DT_FLOAT);
    Tensor wh;
    float* wh_value = new float[288*96];
    wh.SetTensorDesc(tensorDescWh);
    wh.SetData((uint8_t*)wh_value, 288*96*sizeof(float));
    whData.set_attr_value(wh);
    whData.update_output_desc_y(tensorDescWh);

    auto bData = op::Const("bias");
    std::vector<int64_t> dims_b{288,};
    ge::Shape shape_b(dims_b);
    ge::TensorDesc tensorDescB(shape_b, FORMAT_NCHW, DT_FLOAT);
    Tensor bias;
    float* bias_value = new float[288];
    bias.SetTensorDesc(tensorDescB);
    bias.SetData((uint8_t*)bias_value, 288*sizeof(float));
    bData.set_attr_value(bias);
    bData.update_output_desc_y(tensorDescB);

    auto contData = op::Data("cont");
    std::vector<int64_t> dims_cont{20, 1};
    ge::Shape shape_cont(dims_cont);
    ge::TensorDesc tensorDescCont(shape_cont, FORMAT_NCHW,  DT_FLOAT);
    contData.update_input_desc_x(tensorDescCont);
    contData.update_output_desc_y(tensorDescCont);

    auto LSTMOp = op::LSTM("LSTM");
    LSTMOp.set_input_x(xData) \
        .set_input_cont(contData) \
        .set_input_w_x(wxData) \
        .set_input_bias(bData) \
        .set_input_w_h(whData) \
        .set_attr_num_output(72) \
        .set_attr_expose_hidden(false);

    std::vector<Operator> inputs{xData, contData, wxData, bData, whData};
    std::vector<Operator> outputs{LSTMOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ALSTMFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findLstmCell = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BasicLSTMCellV2") {
        findLstmCell = true;
        break;
    }
    }
    EXPECT_EQ(findLstmCell, false);
}

TEST_F(lstm_fusion_test, lstm_fusion_test_2) {
    ge::Graph graph("lstm_fusion_test_2");
    auto xData = op::Data("x");
    std::vector<int64_t> dims_x{1, 1, 512};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    auto x_staticData = op::Data("x_static");
    std::vector<int64_t> dims_x_static{1, 256};
    ge::Shape shape_x_static(dims_x_static);
    ge::TensorDesc tensorDescX_Static(shape_x_static, FORMAT_NCHW,  DT_FLOAT);
    x_staticData.update_input_desc_x(tensorDescX_Static);
    x_staticData.update_output_desc_y(tensorDescX_Static);

    auto h_0Data = op::Data("h_0");
    std::vector<int64_t> dims_h_0{1, 1, 256};
    ge::Shape shape_h_0(dims_h_0);
    ge::TensorDesc tensorDesc_h0(shape_h_0, FORMAT_NCHW,  DT_FLOAT);
    h_0Data.update_input_desc_x(tensorDesc_h0);
    h_0Data.update_output_desc_y(tensorDesc_h0);

    auto c_0Data = op::Data("c_0");
    std::vector<int64_t> dims_c_0{1, 1, 256};
    ge::Shape shape_c_0(dims_c_0);
    ge::TensorDesc tensorDesc_c0(shape_c_0, FORMAT_NCHW,  DT_FLOAT);
    c_0Data.update_input_desc_x(tensorDesc_c0);
    c_0Data.update_output_desc_y(tensorDesc_c0);

    auto wxData = op::Const("w_x");
    std::vector<int64_t> dims_w{1024, 512};
    ge::Shape shape_w(dims_w);
    ge::TensorDesc tensorDescW(shape_w, FORMAT_NCHW,  DT_FLOAT);
    Tensor wx;
    float* wx_value = new float[1024*512];
    wx.SetTensorDesc(tensorDescW);
    wx.SetData((uint8_t*)wx_value, 1024*512*sizeof(float));
    wxData.set_attr_value(wx);
    wxData.update_output_desc_y(tensorDescW);

    auto whData = op::Const("w_h");
    std::vector<int64_t> dims_wh{1024, 256};
    ge::Shape shape_wh(dims_wh);
    ge::TensorDesc tensorDescWh(shape_wh, FORMAT_NCHW, DT_FLOAT);
    Tensor wh;
    float* wh_value = new float[1024*256];
    wh.SetTensorDesc(tensorDescWh);
    wh.SetData((uint8_t*)wh_value, 1024*256*sizeof(float));
    whData.set_attr_value(wh);
    whData.update_output_desc_y(tensorDescWh);

    auto wx_static_Data = op::Const("w_x_static");
    std::vector<int64_t> dims_w_x_static{1024, 256};
    ge::Shape shape_w_x_static(dims_w_x_static);
    ge::TensorDesc tensorDesc_x_static(shape_w_x_static, FORMAT_NCHW, DT_FLOAT);
    Tensor w_x_static;
    float* w_x_static_value = new float[1024*256];
    w_x_static.SetTensorDesc(tensorDesc_x_static);
    w_x_static.SetData((uint8_t*)w_x_static_value, 1024*256*sizeof(float));
    wx_static_Data.set_attr_value(w_x_static);
    wx_static_Data.update_output_desc_y(tensorDesc_x_static);

    auto bData = op::Const("bias");
    std::vector<int64_t> dims_b{1024,};
    ge::Shape shape_b(dims_b);
    ge::TensorDesc tensorDescB(shape_b, FORMAT_NCHW, DT_FLOAT);
    Tensor bias;
    float* bias_value = new float[1024];
    bias.SetTensorDesc(tensorDescB);
    bias.SetData((uint8_t*)bias_value, 1024*sizeof(float));
    bData.set_attr_value(bias);
    bData.update_output_desc_y(tensorDescB);

    auto contData = op::Data("cont");
    std::vector<int64_t> dims_cont{1, 1};
    ge::Shape shape_cont(dims_cont);
    ge::TensorDesc tensorDescCont(shape_cont, FORMAT_NCHW,  DT_FLOAT);
    contData.update_input_desc_x(tensorDescCont);
    contData.update_output_desc_y(tensorDescCont);

    auto LSTMOp = op::LSTM("LSTM");
    LSTMOp.set_input_x(xData) \
        .set_input_cont(contData) \
        .set_input_w_x(x_staticData) \
        .set_input_bias(h_0Data) \
        .set_input_w_h(c_0Data) \
        .set_input_x_static(wxData) \
        .set_input_h_0(bData) \
        .set_input_c_0(wx_static_Data) \
        .set_input_w_x_static(whData) \
        .set_attr_num_output(72) \
        .set_attr_expose_hidden(true);

    std::vector<Operator> inputs{xData, contData, x_staticData, h_0Data, c_0Data};
    std::vector<Operator> outputs{LSTMOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ALSTMFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findLstmCell = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "BasicLSTMCellV2") {
            findLstmCell = true;
            break;
        }
    }
    EXPECT_EQ(findLstmCell, true);
}
}