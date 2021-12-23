#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "nn_calculation_ops.h"
#include "gtest/gtest.h"

using namespace ge;
using namespace op;

namespace fe {
class dynamic_lstm_v2_fusion_test : public testing::Test {
  protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_lstm_v2_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dynamic_lstm_v2_fusion_test TearDown" << std::endl;
    }
};

void BuildGraphForSplit(ge::ComputeGraphPtr &parent_graph, ge::ComputeGraphPtr &sub_graph) {

    ge::GeShape wx_shape({1024, 512});
    ge::GeTensorDesc wx_desc(wx_shape, ge::FORMAT_ND, ge::DT_FLOAT);
    wx_desc.SetOriginFormat(ge::FORMAT_ND);
    wx_desc.SetOriginDataType(ge::DT_FLOAT);
    wx_desc.SetOriginShape(wx_shape);
    ge::OpDescPtr x1 = std::make_shared<ge::OpDesc>("x1", "Const");
    ge::OpDescPtr func = std::make_shared<ge::OpDesc>("func", "PartitionedCall");
    ge::OpDescPtr output = std::make_shared<ge::OpDesc>("output", "NetOutput");
    x1->AddInputDesc(wx_desc);
    x1->AddOutputDesc(wx_desc);
    func->AddOutputDesc(wx_desc);
    func->AddInputDesc(wx_desc);
    output->AddInputDesc(wx_desc);
    ge::AttrUtils::SetBool(x1, "const_adjust_flag", true);

    parent_graph = std::make_shared<ge::ComputeGraph>("parentgraph");
    ge::NodePtr x1_node = parent_graph->AddNode(x1);
    ge::NodePtr func_node = parent_graph->AddNode(func);
    ge::NodePtr output_node = parent_graph->AddNode(output);
    ge::GraphUtils::AddEdge(x1_node->GetOutDataAnchor(0), func_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(func_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));

    float *wx_value = new float[1024 * 768];
    ge::GeTensorPtr weightTensor = nullptr;
    weightTensor = std::make_shared<GeTensor>(wx_desc, reinterpret_cast<uint8_t *>(wx_value), 1024 * 768 * sizeof(float));
    ge::OpDescUtils::SetWeights(x1_node, {weightTensor});

    ge::GeShape input_shape({75, 1, 512});
    ge::GeTensorDesc x_desc(input_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    x_desc.SetOriginFormat(ge::FORMAT_NCHW);
    x_desc.SetOriginDataType(ge::DT_FLOAT);
    x_desc.SetOriginShape(input_shape);

    ge::GeShape wh_shape({1024, 256});
    ge::GeTensorDesc wh_desc(wh_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    wh_desc.SetOriginFormat(ge::FORMAT_NCHW);
    wh_desc.SetOriginDataType(ge::DT_FLOAT);
    wh_desc.SetOriginShape(wh_shape);

    ge::GeShape bias_shape({1024});
    ge::GeTensorDesc bias_desc(bias_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    bias_desc.SetOriginFormat(ge::FORMAT_NCHW);
    bias_desc.SetOriginDataType(ge::DT_FLOAT);
    bias_desc.SetOriginShape(bias_shape);

    ge::GeShape output_shape({75, 1, 1024});
    ge::GeTensorDesc y_desc(output_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    y_desc.SetOriginFormat(ge::FORMAT_NCHW);
    y_desc.SetOriginDataType(ge::DT_FLOAT);
    y_desc.SetOriginShape(output_shape);

    ge::GeShape cont_shape({75, 1});
    ge::GeTensorDesc cont_desc(cont_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    cont_desc.SetOriginFormat(ge::FORMAT_NCHW);
    cont_desc.SetOriginDataType(ge::DT_FLOAT);
    cont_desc.SetOriginShape(cont_shape);

    ge::OpDescPtr x = std::make_shared<ge::OpDesc>("x", "Data");
    ge::OpDescPtr wx = std::make_shared<ge::OpDesc>("w_x", "Data");
    ge::OpDescPtr wh = std::make_shared<ge::OpDesc>("w_h", "Const");
    ge::OpDescPtr bias = std::make_shared<ge::OpDesc>("bias", "Const");
    ge::OpDescPtr cont = std::make_shared<ge::OpDesc>("cont", "Data");
    ge::OpDescPtr lstm = std::make_shared<ge::OpDesc>("lstm", "LSTM");
    ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

    x->AddOutputDesc(x_desc);
    wx->AddOutputDesc(wx_desc);
    wh->AddOutputDesc(wh_desc);
    bias->AddOutputDesc(bias_desc);
    cont->AddOutputDesc(cont_desc);
    lstm->AddInputDesc("x", x_desc);
    lstm->AddInputDesc("cont", cont_desc);
    lstm->AddInputDesc("w_x", wx_desc);
    lstm->AddInputDesc("bias", bias_desc);
    lstm->AddInputDesc("w_h", wh_desc);
    lstm->AddOutputDesc("h", y_desc);
    lstm->AddOutputDesc("h_t", y_desc);
    lstm->AddOutputDesc("c_t", y_desc);

    netoutput->AddInputDesc(y_desc);
    ge::AttrUtils::SetInt(lstm, "num_output", 256);
    ge::AttrUtils::SetBool(lstm, "expose_hidden", false);

    sub_graph = std::make_shared<ge::ComputeGraph>("subgraph");
    ge::NodePtr x_node = sub_graph->AddNode(x);
    ge::NodePtr cont_node = sub_graph->AddNode(cont);
    ge::NodePtr wx_node = sub_graph->AddNode(wx);
    ge::NodePtr bias_node = sub_graph->AddNode(bias);
    ge::NodePtr wh_node = sub_graph->AddNode(wh);
    ge::NodePtr lstm_node = sub_graph->AddNode(lstm);
    ge::NodePtr netoutput_node = sub_graph->AddNode(netoutput);
    ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), lstm_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(cont_node->GetOutDataAnchor(0), lstm_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(wx_node->GetOutDataAnchor(0), lstm_node->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(bias_node->GetOutDataAnchor(0), lstm_node->GetInDataAnchor(3));
    ge::GraphUtils::AddEdge(wh_node->GetOutDataAnchor(0), lstm_node->GetInDataAnchor(4));
    ge::GraphUtils::AddEdge(lstm_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(lstm_node->GetOutDataAnchor(1), netoutput_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(lstm_node->GetOutDataAnchor(2), netoutput_node->GetInDataAnchor(0));
    ge::AttrUtils::SetInt(wx_node->GetOpDesc(), "_parent_node_index", 0);

    func_node->GetOpDesc()->AddSubgraphName("f");
    func_node->GetOpDesc()->SetSubgraphInstanceName(0, sub_graph->GetName());
    sub_graph->SetParentNode(func_node);
    sub_graph->SetParentGraph(parent_graph);
    parent_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
}

TEST_F(dynamic_lstm_v2_fusion_test, dynamic_lstm_v2_fusion_test_1) {
    ge::Graph graph("dynamic_lstm_v2_fusion_test_1");

    auto xData = op::Data("xData");
    std::vector<int64_t> dims_x{75, 1, 512};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW, DT_FLOAT);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    auto wxData = op::Const("w_x");
    std::vector<int64_t> dims_w{1024, 512};
    ge::Shape shape_w(dims_w);
    ge::TensorDesc tensorDescW(shape_w, FORMAT_NCHW, DT_FLOAT);
    Tensor wx;
    float *wx_value = new float[1024 * 512];
    wx.SetTensorDesc(tensorDescW);
    wx.SetData((uint8_t *)wx_value, 1024 * 512 * sizeof(float));
    wxData.set_attr_value(wx);
    wxData.update_output_desc_y(tensorDescW);

    auto whData = op::Const("w_h");
    std::vector<int64_t> dims_wh{1024, 256};
    ge::Shape shape_wh(dims_wh);
    ge::TensorDesc tensorDescWh(shape_wh, FORMAT_NCHW, DT_FLOAT);
    Tensor wh;
    float *wh_value = new float[1024 * 256];
    wh.SetTensorDesc(tensorDescWh);
    wh.SetData((uint8_t *)wh_value, 1024 * 256 * sizeof(float));
    whData.set_attr_value(wh);
    whData.update_output_desc_y(tensorDescWh);

    auto bData = op::Const("bias");
    std::vector<int64_t> dims_b{1024};
    ge::Shape shape_b(dims_b);
    ge::TensorDesc tensorDescB(shape_b, FORMAT_NCHW, DT_FLOAT);
    Tensor bias;
    float *bias_value = new float[1024];
    bias.SetTensorDesc(tensorDescB);
    bias.SetData((uint8_t *)bias_value, 1024 * sizeof(float));
    bData.set_attr_value(bias);
    bData.update_output_desc_y(tensorDescB);

    auto contData = op::Data("cont");
    std::vector<int64_t> dims_cont{75, 1};
    ge::Shape shape_cont(dims_cont);
    ge::TensorDesc tensorDescCont(shape_cont, FORMAT_NCHW, DT_FLOAT);
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
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DynamicLSTMV2") {
            findTranspose = true;
            break;
        }
    }
    EXPECT_EQ(findTranspose, true);
}
TEST_F(dynamic_lstm_v2_fusion_test, input_weight_parent_graph_test) {
    ge::ComputeGraphPtr parent_graph;
    ge::ComputeGraphPtr sub_graph;
    BuildGraphForSplit(parent_graph, sub_graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("DynamicLSTMFusionPass", fe::BUILT_IN_GRAPH_PASS, *sub_graph);
    bool find_dynamicLSTM = false;
    for (auto node : sub_graph->GetAllNodes()) {
        if (node->GetType() == "DynamicLSTMV2") {
            find_dynamicLSTM = true;
            break;
        }
    }
    EXPECT_EQ(find_dynamicLSTM, true);
}
TEST_F(dynamic_lstm_v2_fusion_test, dynamic_lstm_v2_fusion_test_2) {
    ge::Graph graph("dynamic_lstm_v2_fusion_test_1");

    auto xData = op::Data("xData");
    std::vector<int64_t> dims_x{75, 1, 512};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW, DT_FLOAT16);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    auto wxData = op::Const("w_x");
    std::vector<int64_t> dims_w{1024, 512};
    ge::Shape shape_w(dims_w);
    ge::TensorDesc tensorDescW(shape_w, FORMAT_NCHW, DT_FLOAT);
    Tensor wx;
    float *wx_value = new float[1024 * 512];
    wx.SetTensorDesc(tensorDescW);
    wx.SetData((uint8_t *)wx_value, 1024 * 512 * sizeof(float));
    wxData.set_attr_value(wx);
    wxData.update_output_desc_y(tensorDescW);

    auto whData = op::Const("w_h");
    std::vector<int64_t> dims_wh{1024, 256};
    ge::Shape shape_wh(dims_wh);
    ge::TensorDesc tensorDescWh(shape_wh, FORMAT_NCHW, DT_FLOAT);
    Tensor wh;
    float *wh_value = new float[1024 * 256];
    wh.SetTensorDesc(tensorDescWh);
    wh.SetData((uint8_t *)wh_value, 1024 * 256 * sizeof(float));
    whData.set_attr_value(wh);
    whData.update_output_desc_y(tensorDescWh);

    auto bData = op::Const("bias");
    std::vector<int64_t> dims_b{1024};
    ge::Shape shape_b(dims_b);
    ge::TensorDesc tensorDescB(shape_b, FORMAT_NCHW, DT_FLOAT);
    Tensor bias;
    float *bias_value = new float[1024];
    bias.SetTensorDesc(tensorDescB);
    bias.SetData((uint8_t *)bias_value, 1024 * sizeof(float));
    bData.set_attr_value(bias);
    bData.update_output_desc_y(tensorDescB);

    auto contData = op::Data("cont");
    std::vector<int64_t> dims_cont{75, 1};
    ge::Shape shape_cont(dims_cont);
    ge::TensorDesc tensorDescCont(shape_cont, FORMAT_NCHW, DT_FLOAT);
    contData.update_input_desc_x(tensorDescCont);
    contData.update_output_desc_y(tensorDescCont);

    auto xstaticData = op::Const("xstatic");
    std::vector<int64_t> dims_xstatic{1024, 256};
    ge::Shape shape_xstatic(dims_xstatic);
    ge::TensorDesc tensorDescxstatic(shape_xstatic, FORMAT_NCHW, DT_FLOAT16);
    Tensor xstatic;
    float *xstatic_value = new float[1024 * 256];
    xstatic.SetTensorDesc(tensorDescxstatic);
    xstatic.SetData((uint8_t *)xstatic_value, 1024 * 256 * sizeof(float));
    xstaticData.set_attr_value(xstatic);
    xstaticData.update_output_desc_y(tensorDescxstatic);

    auto h0Data = op::Const("h0");
    std::vector<int64_t> dims_h0{1024, 256};
    ge::Shape shape_h0(dims_h0);
    ge::TensorDesc tensorDesch0(shape_h0, FORMAT_NCHW, DT_FLOAT16);
    Tensor h0;
    float *h0_value = new float[1024 * 256];
    h0.SetTensorDesc(tensorDescWh);
    h0.SetData((uint8_t *)h0_value, 1024 * 256 * sizeof(float));
    h0Data.set_attr_value(h0);
    h0Data.update_output_desc_y(tensorDesch0);

    auto c0Data = op::Const("c0");
    std::vector<int64_t> dims_c0{1024, 256};
    ge::Shape shape_c0(dims_c0);
    ge::TensorDesc tensorDescc0(shape_c0, FORMAT_NCHW, DT_FLOAT16);
    Tensor c0;
    float *c0_value = new float[1024 * 256];
    c0.SetTensorDesc(tensorDescc0);
    c0.SetData((uint8_t *)c0_value, 1024 * 256 * sizeof(float));
    c0Data.set_attr_value(c0);
    c0Data.update_output_desc_y(tensorDescc0);

    auto wxstaticData = op::Const("wxstatic");
    std::vector<int64_t> dims_wxstatic{1024, 256};
    ge::Shape shape_wxstatic(dims_wxstatic);
    ge::TensorDesc tensorDescwxstatic(shape_wxstatic, FORMAT_NCHW, DT_FLOAT16);
    Tensor wxstatic;
    float *wxstatic_value = new float[1024 * 256];
    wxstatic.SetTensorDesc(tensorDescwxstatic);
    wxstatic.SetData((uint8_t *)wxstatic_value, 1024 * 256 * sizeof(float));
    wxstaticData.set_attr_value(wxstatic);
    wxstaticData.update_output_desc_y(tensorDescwxstatic);

    auto LSTMOp = op::LSTM("LSTM");
    LSTMOp.set_input_x(xData)
        .set_input_cont(contData)
        .set_input_w_x(wxData)
        .set_input_bias(bData)
        .set_input_w_h(whData)
        .set_input_x_static(xstaticData)
        .set_input_h_0(h0Data)
        .set_input_c_0(c0Data)
        .set_input_w_x_static(wxstaticData)
        .set_attr_num_output(256)
        .set_attr_expose_hidden(false);

    std::vector<Operator> inputs{xData, contData, wxData, bData, whData,xstaticData,h0Data,c0Data,wxstaticData};
    std::vector<Operator> outputs{LSTMOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("DynamicLSTMFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DynamicLSTMV2") {
            findTranspose = true;
            break;
        }
    }
    EXPECT_EQ(findTranspose, true);
}
} // namespace fe