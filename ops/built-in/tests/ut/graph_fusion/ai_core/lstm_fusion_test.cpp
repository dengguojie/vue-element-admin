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

void BuildGraphForSubAndParent(ge::ComputeGraphPtr &parent_graph, ge::ComputeGraphPtr &sub_graph) {
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

    ge::GeShape input_shape({1, 1, 512});
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

    ge::GeShape output_shape({1, 1, 1024});
    ge::GeTensorDesc y_desc(output_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    y_desc.SetOriginFormat(ge::FORMAT_NCHW);
    y_desc.SetOriginDataType(ge::DT_FLOAT);
    y_desc.SetOriginShape(output_shape);

    ge::GeShape cont_shape({1, 1});
    ge::GeTensorDesc cont_desc(cont_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    cont_desc.SetOriginFormat(ge::FORMAT_NCHW);
    cont_desc.SetOriginDataType(ge::DT_FLOAT);
    cont_desc.SetOriginShape(cont_shape);

//    auto x_staticData = op::Data("x_static");
    std::vector<int64_t> dims_x_static{1, 256};
    ge::GeShape shape_x_static(dims_x_static);
    ge::GeTensorDesc tensorDescX_Static(shape_x_static, ge::FORMAT_NCHW,  ge::DT_FLOAT);
    tensorDescX_Static.SetOriginFormat(ge::FORMAT_NCHW);
    tensorDescX_Static.SetOriginDataType(ge::DT_FLOAT);
    tensorDescX_Static.SetOriginShape(shape_x_static);

//    auto wx_static_Data = op::Const("w_x_static");

    std::vector<int64_t> dims_w_x_static{1024, 256};
    ge::GeShape shape_w_x_static(dims_w_x_static);
    ge::GeTensorDesc tensorDesc_x_static(shape_w_x_static, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensorDesc_x_static.SetOriginFormat(ge::FORMAT_NCHW);
    tensorDesc_x_static.SetOriginDataType(ge::DT_FLOAT);
    tensorDesc_x_static.SetOriginShape(shape_w_x_static);

    ge::OpDescPtr x = std::make_shared<ge::OpDesc>("x", "Data");
    ge::OpDescPtr wx = std::make_shared<ge::OpDesc>("w_x", "Data");
    ge::OpDescPtr wh = std::make_shared<ge::OpDesc>("w_h", "Const");
    ge::OpDescPtr bias = std::make_shared<ge::OpDesc>("bias", "Const");
    ge::OpDescPtr cont = std::make_shared<ge::OpDesc>("cont", "Data");

    ge::OpDescPtr wxStatic = std::make_shared<ge::OpDesc>("w_x_static", "Const");
    ge::OpDescPtr xStatic = std::make_shared<ge::OpDesc>("x_static", "Data");

    ge::OpDescPtr lstm = std::make_shared<ge::OpDesc>("lstm", "LSTM");
    ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

    x->AddOutputDesc(x_desc);
    wx->AddOutputDesc(wx_desc);
    wh->AddInputDesc(wh_desc);
    wh->AddOutputDesc(wh_desc);
    bias->AddInputDesc(bias_desc);
    bias->AddOutputDesc(bias_desc);
    cont->AddOutputDesc(cont_desc);

    wxStatic->AddInputDesc(tensorDesc_x_static);
    wxStatic->AddOutputDesc(tensorDesc_x_static);
    xStatic->AddOutputDesc(tensorDescX_Static);

    lstm->AddInputDesc("x", x_desc);
    lstm->AddInputDesc("cont", cont_desc);
    lstm->AddInputDesc("w_x", tensorDescX_Static);
    lstm->AddInputDesc("bias", wx_desc);
    lstm->AddInputDesc("w_h", bias_desc);
    lstm->AddInputDesc("x_static", tensorDesc_x_static);
    lstm->AddInputDesc("w_x_static", wh_desc);

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
    ge::NodePtr wxStatic_node = sub_graph->AddNode(wxStatic);

    float *wstatic_value = new float[1024 * 256];
    ge::GeTensorPtr weightStaticTensor = nullptr;
    weightStaticTensor = std::make_shared<GeTensor>(tensorDesc_x_static,
                                                    reinterpret_cast<uint8_t *>(wstatic_value),
                                                    1024 * 256 * sizeof(float));
    ge::OpDescUtils::SetWeights(wxStatic_node, {weightStaticTensor});

    float *bias_value = new float[1024];
    ge::GeTensorPtr biasTensor = nullptr;
    biasTensor = std::make_shared<GeTensor>(bias_desc,
                                            reinterpret_cast<uint8_t *>(bias_value),
                                            1024 * sizeof(float));
    ge::OpDescUtils::SetWeights(bias_node, {biasTensor});

    float *wh_value = new float[1024 * 256];
    ge::GeTensorPtr whTensor = nullptr;
    whTensor = std::make_shared<GeTensor>(wh_desc,
                                          reinterpret_cast<uint8_t *>(wh_value),
                                          1024 * 256 * sizeof(float));
    ge::OpDescUtils::SetWeights(wh_node, {whTensor});

    ge::NodePtr xStatic_node = sub_graph->AddNode(xStatic);
    ge::NodePtr lstm_node = sub_graph->AddNode(lstm);
    ge::NodePtr netoutput_node = sub_graph->AddNode(netoutput);
    ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), lstm_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(cont_node->GetOutDataAnchor(0), lstm_node->GetInDataAnchor(1));

    ge::GraphUtils::AddEdge(xStatic_node->GetOutDataAnchor(0), lstm_node->GetInDataAnchor(2));

    ge::GraphUtils::AddEdge(wx_node->GetOutDataAnchor(0), lstm_node->GetInDataAnchor(3));
    ge::GraphUtils::AddEdge(bias_node->GetOutDataAnchor(0), lstm_node->GetInDataAnchor(4));

    ge::GraphUtils::AddEdge(wxStatic_node->GetOutDataAnchor(0), lstm_node->GetInDataAnchor(5));

    ge::GraphUtils::AddEdge(wh_node->GetOutDataAnchor(0), lstm_node->GetInDataAnchor(6));

    ge::GraphUtils::AddEdge(lstm_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(lstm_node->GetOutDataAnchor(1), netoutput_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(lstm_node->GetOutDataAnchor(2), netoutput_node->GetInDataAnchor(0));
    ge::AttrUtils::SetInt(wx_node->GetOpDesc(), "_parent_node_index", 0);
    int32_t input_size = lstm_node->GetInDataNodes().size();
    func_node->GetOpDesc()->AddSubgraphName("f");
    func_node->GetOpDesc()->SetSubgraphInstanceName(0, sub_graph->GetName());
    sub_graph->SetParentNode(func_node);
    sub_graph->SetParentGraph(parent_graph);
    parent_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
}

TEST_F(lstm_fusion_test, lstm_fusion_test_3) {
    ge::ComputeGraphPtr parent_graph;
    ge::ComputeGraphPtr sub_graph;
    BuildGraphForSubAndParent(parent_graph, sub_graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ALSTMFusionPass", fe::BUILT_IN_GRAPH_PASS, *sub_graph);
    bool findLstmCell = false;
    for (auto node : sub_graph->GetAllNodes()) {
        if (node->GetType() == "BasicLSTMCellV2") {
            findLstmCell = true;
            break;
        }
    }
    EXPECT_EQ(findLstmCell, true);
}


}
