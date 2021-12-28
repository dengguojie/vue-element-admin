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

class lstmp_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "lstmp_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "lstmp_fusion_test TearDown" << std::endl;
    }
};

TEST_F(lstmp_fusion_test, lstmp_fusion_test_1) {
    ge::Graph graph("lstmp_fusion_test");

    auto xData = op::Data("xData");
    std::vector<int64_t> dims_x{16,16,16};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND, DT_FLOAT);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    auto mData = op::Data("mData");
    std::vector<int64_t> dims_m{16,};
    ge::Shape shape_m(dims_m);
    ge::TensorDesc tensorDescM(shape_m, FORMAT_ND, DT_FLOAT);
    mData.update_input_desc_x(tensorDescM);
    mData.update_output_desc_y(tensorDescM);
    
    auto rData = op::Data("rData");
    std::vector<int64_t> dims_r{16,16};
    ge::Shape shape_r(dims_r);
    ge::TensorDesc tensorDescR(shape_r, FORMAT_ND, DT_FLOAT);
    rData.update_input_desc_x(tensorDescR);
    rData.update_output_desc_y(tensorDescR);

    auto cData = op::Data("cData");
    std::vector<int64_t> dims_c{16,16};
    ge::Shape shape_c(dims_c);
    ge::TensorDesc tensorDescC(shape_c, FORMAT_ND, DT_FLOAT);
    cData.update_input_desc_x(tensorDescC);
    cData.update_output_desc_y(tensorDescC);
    
    auto wxData = op::Const("wxData");
    std::vector<int64_t> dims_wx{64,16};
    ge::Shape shape_wx(dims_wx);
    ge::TensorDesc tensorDescWX(shape_wx, FORMAT_HWCN, DT_FLOAT);

    Tensor wx_tensor;
    float *wx_tensor_value = new float[64*16];
    wx_tensor.SetTensorDesc(tensorDescWX);
    wx_tensor.SetData((uint8_t*)wx_tensor_value, 64*16*sizeof(float));
    wxData.set_attr_value(wx_tensor);

    wxData.update_output_desc_y(tensorDescWX);

    auto wrData = op::Const("wrData");
    std::vector<int64_t> dims_wr{64,16};
    ge::Shape shape_wr(dims_wr);
    ge::TensorDesc tensorDescWR(shape_wr, FORMAT_HWCN, DT_FLOAT);

    wrData.set_attr_value(wx_tensor);
    wrData.update_output_desc_y(tensorDescWR);

    auto bData = op::Const("bData");
    std::vector<int64_t> dims_b{64,};
    ge::Shape shape_b(dims_b);
    ge::TensorDesc tensorDescB(shape_b, FORMAT_ND, DT_FLOAT);

    Tensor b_tensor;
    float *b_tensor_value = new float[64];
    b_tensor.SetTensorDesc(tensorDescB);
    b_tensor.SetData((uint8_t*)b_tensor_value, 64*sizeof(float));
    bData.set_attr_value(b_tensor);
    bData.update_output_desc_y(tensorDescB);

    auto wpData = op::Data("wpData");
    std::vector<int64_t> dims_wp{16,16};
    ge::Shape shape_wp(dims_wp);
    ge::TensorDesc tensorDescWP(shape_wp, FORMAT_ND, DT_FLOAT);
    wpData.update_input_desc_x(tensorDescWP);
    wpData.update_output_desc_y(tensorDescWP);
    
    auto LSTMPOp = op::LSTMP("LSTMP");
    LSTMPOp.set_input_x(xData)
         .set_input_wx(wxData)
         .set_input_bias(bData)
         .set_input_wr(wrData)
         .set_input_project(wpData)
         .set_input_real_mask(mData)
         .set_input_init_h(rData)
         .set_input_init_c(cData);

    std::vector<Operator> inputs{xData, wxData, bData, wrData, wpData, mData, rData, cData};
    std::vector<Operator> outputs{LSTMPOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LSTMPFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DynamicRNNV3") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, true);
}

TEST_F(lstmp_fusion_test, lstmp_fusion_test_2) {
    ge::Graph graph("lstmp_fusion_test");

    auto xData = op::Data("xData");
    std::vector<int64_t> dims_x{16,16,16};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND, DT_FLOAT);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    auto wxData = op::Const("wxData");
    std::vector<int64_t> dims_wx{64,16};
    ge::Shape shape_wx(dims_wx);
    ge::TensorDesc tensorDescWX(shape_wx, FORMAT_HWCN, DT_FLOAT);

    Tensor wx_tensor;
    float *wx_tensor_value = new float[64*16];
    wx_tensor.SetTensorDesc(tensorDescWX);
    wx_tensor.SetData((uint8_t*)wx_tensor_value, 64*16*sizeof(float));

    wxData.set_attr_value(wx_tensor);
    wxData.update_output_desc_y(tensorDescWX);

    auto wrData = op::Const("wrData");
    std::vector<int64_t> dims_wr{64,16};
    ge::Shape shape_wr(dims_wr);
    ge::TensorDesc tensorDescWR(shape_wr, FORMAT_HWCN, DT_FLOAT);

    wrData.set_attr_value(wx_tensor);
    wrData.update_output_desc_y(tensorDescWR);

    auto bData = op::Const("bData");
    std::vector<int64_t> dims_b{64,};
    ge::Shape shape_b(dims_b);
    ge::TensorDesc tensorDescB(shape_b, FORMAT_ND, DT_FLOAT);

    Tensor b_tensor;
    float *b_tensor_value = new float[64];
    b_tensor.SetTensorDesc(tensorDescB);
    b_tensor.SetData((uint8_t*)b_tensor_value, 64*sizeof(float));
    bData.set_attr_value(b_tensor);
    bData.update_output_desc_y(tensorDescB);

    auto wpData = op::Data("wpData");
    std::vector<int64_t> dims_wp{16,16};
    ge::Shape shape_wp(dims_wp);
    ge::TensorDesc tensorDescWP(shape_wp, FORMAT_ND, DT_FLOAT);
    wpData.update_input_desc_x(tensorDescWP);
    wpData.update_output_desc_y(tensorDescWP);
    
    auto LSTMPOp = op::LSTMP("LSTMP");
    LSTMPOp.set_input_x(xData)
         .set_input_wx(wxData)
         .set_input_bias(bData)
         .set_input_wr(wrData)
         .set_input_project(wpData);

    std::vector<Operator> inputs{xData, wxData, bData, wrData, wpData};
    std::vector<Operator> outputs{LSTMPOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LSTMPFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DynamicRNNV3") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, true);
}
