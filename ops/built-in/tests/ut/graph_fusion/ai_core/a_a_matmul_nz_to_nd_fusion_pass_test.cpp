#include "gtest/gtest.h"
#include "array_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "transformation_ops.h"
#include "matrix_calculation_ops.h"
#include "framework/common/types.h"
#define private public
#define protected public
#include "fusion_pass_test_utils.h"
#include "fusion_pass_test_slice_utils.h"


using namespace ge;
using namespace op;

class a_a_matmul_nz_to_nd_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "a_a_matmul_nz_to_nd_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "a_a_matmul_nz_to_nd_fusion_pass_test TearDown" << std::endl;
    }
};

// Data(nd,required) Data(nd,required) Data(nd,optional)
//    \                 /                    /
//  TransData(nd->Nz) TransData(nd->Nz)     /
//      \             /                    /
//      MatMul/MatMulV2/BatchMatMul/BatchMatMulV2(Nz)
//                  |
//               TransData

#define DESC_DATA(name, shape_in, format_in, shape_out, format_out, dtype) \
    ge::GeTensorDesc desc_##name(shape_out, format_out, dtype); \
    desc_##name.SetOriginFormat(format_in);                \
    desc_##name.SetOriginShape(shape_in)

TEST_F(a_a_matmul_nz_to_nd_fusion_pass_test, batchmatmul_without_bias) {
    ge::Graph graph("batchmatmul_without_bias");

    DESC_DATA(data_a, ge::GeShape({16, 512, 1024}), FORMAT_ND, ge::GeShape({16, 512, 1024}), FORMAT_ND, DT_FLOAT16);
    DESC_DATA(data_b, ge::GeShape({16, 1024, 4096}), FORMAT_ND, ge::GeShape({16, 512, 1024}), FORMAT_ND, DT_FLOAT16);

    DESC_DATA(trans_a, ge::GeShape({16, 512, 1024}), FORMAT_ND, ge::GeShape({16, 64, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    DESC_DATA(trans_b, ge::GeShape({16, 1024, 4096}), FORMAT_ND, ge::GeShape({16, 256, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);

    DESC_DATA(batchmatmul_y, ge::GeShape({16, 512, 4096}), FORMAT_ND, ge::GeShape({16, 256, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    DESC_DATA(trans_c, ge::GeShape({16, 512, 1024}), FORMAT_ND, ge::GeShape({16, 512, 1024}), FORMAT_ND, DT_FLOAT16);

    ge::OpDescPtr data_a = std::make_shared<ge::OpDesc>("data_a", "Data");
    ge::OpDescPtr data_b = std::make_shared<ge::OpDesc>("data_b", "Data");
    ge::OpDescPtr trans_a = std::make_shared<ge::OpDesc>("trans_a", "TransData");
    ge::OpDescPtr trans_b = std::make_shared<ge::OpDesc>("trans_b", "TransData");
    ge::OpDescPtr batchmatmul = std::make_shared<ge::OpDesc>("batchmatmul", "BatchMatMul");
    ge::OpDescPtr trans_c = std::make_shared<ge::OpDesc>("trans_c", "TransData");
    ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

    data_a->AddOutputDesc(desc_data_a);
    data_b->AddOutputDesc(desc_data_b);
    trans_a->AddInputDesc(desc_data_a);
    trans_a->AddOutputDesc(desc_trans_a);
    trans_b->AddInputDesc(desc_data_b);
    trans_b->AddOutputDesc(desc_trans_b);
    batchmatmul->AddInputDesc("x1", desc_trans_a);
    batchmatmul->AddInputDesc("x2", desc_trans_b);
    batchmatmul->AddOutputDesc(desc_batchmatmul_y);
    trans_c->AddInputDesc(desc_batchmatmul_y);
    trans_c->AddOutputDesc(desc_trans_c);
    netoutput->AddInputDesc(desc_trans_c);

    ge::AttrUtils::SetStr(trans_a, "src_format", "ND");
    ge::AttrUtils::SetStr(trans_a, "dst_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_b, "src_format", "ND");
    ge::AttrUtils::SetStr(trans_b, "dst_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_c, "src_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_c, "dst_format", "ND");
    ge::AttrUtils::SetBool(batchmatmul, "adj_x1", false);
    ge::AttrUtils::SetBool(batchmatmul, "adj_x2", false);

    ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("subgraph");
    ge::NodePtr data_a_node = compute_graph_ptr->AddNode(data_a);
    ge::NodePtr data_b_node = compute_graph_ptr->AddNode(data_b);
    ge::NodePtr trans_a_node = compute_graph_ptr->AddNode(trans_a);
    ge::NodePtr trans_b_node = compute_graph_ptr->AddNode(trans_b);
    ge::NodePtr batchmatmul_node = compute_graph_ptr->AddNode(batchmatmul);
    ge::NodePtr trans_c_node = compute_graph_ptr->AddNode(trans_c);
    ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

    ge::GraphUtils::AddEdge(data_a_node->GetOutDataAnchor(0), trans_a_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_b_node->GetOutDataAnchor(0), trans_b_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_a_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_b_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(batchmatmul_node->GetOutDataAnchor(0), trans_c_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_c_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

    fe::FusionPassTestUtils::RunGraphFusionPass("AAMatMulNzToNdFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool res = true;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
      if (node->GetType() == "TransData") {
        res = false;
      }
    }
    EXPECT_EQ(res, true);
}

TEST_F(a_a_matmul_nz_to_nd_fusion_pass_test, matmul_without_bias) {
    ge::Graph graph("matmul_without_bias");

    DESC_DATA(data_a, ge::GeShape({512, 1024}), FORMAT_ND, ge::GeShape({512, 1024}), FORMAT_ND, DT_FLOAT16);
    DESC_DATA(data_b, ge::GeShape({1024, 4096}), FORMAT_ND, ge::GeShape({1024, 4096}), FORMAT_ND, DT_FLOAT16);

    DESC_DATA(trans_a, ge::GeShape({512, 1024}), FORMAT_ND, ge::GeShape({64, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    DESC_DATA(trans_b, ge::GeShape({1024, 4096}), FORMAT_ND, ge::GeShape({256, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);

    DESC_DATA(matmul_y, ge::GeShape({512, 4096}), FORMAT_ND, ge::GeShape({256, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    DESC_DATA(trans_c, ge::GeShape({512, 4096}), FORMAT_ND, ge::GeShape({512, 4096}), FORMAT_ND, DT_FLOAT16);

    ge::OpDescPtr data_a = std::make_shared<ge::OpDesc>("data_a", "Data");
    ge::OpDescPtr data_b = std::make_shared<ge::OpDesc>("data_b", "Data");
    ge::OpDescPtr trans_a = std::make_shared<ge::OpDesc>("trans_a", "TransData");
    ge::OpDescPtr trans_b = std::make_shared<ge::OpDesc>("trans_b", "TransData");
    ge::OpDescPtr matmul = std::make_shared<ge::OpDesc>("matmul", "MatMul");
    ge::OpDescPtr trans_c = std::make_shared<ge::OpDesc>("trans_c", "TransData");
    ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

    data_a->AddOutputDesc(desc_data_a);
    data_b->AddOutputDesc(desc_data_b);
    trans_a->AddInputDesc(desc_data_a);
    trans_a->AddOutputDesc(desc_trans_a);
    trans_b->AddInputDesc(desc_data_b);
    trans_b->AddOutputDesc(desc_trans_b);
    matmul->AddInputDesc("x1", desc_trans_a);
    matmul->AddInputDesc("x2", desc_trans_b);
    matmul->AddOutputDesc(desc_matmul_y);
    trans_c->AddInputDesc(desc_matmul_y);
    trans_c->AddOutputDesc(desc_trans_c);
    netoutput->AddInputDesc(desc_trans_c);

    ge::AttrUtils::SetStr(trans_a, "src_format", "ND");
    ge::AttrUtils::SetStr(trans_a, "dst_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_b, "src_format", "ND");
    ge::AttrUtils::SetStr(trans_b, "dst_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_c, "src_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_c, "dst_format", "ND");
    ge::AttrUtils::SetBool(matmul, "transpose_x1", false);
    ge::AttrUtils::SetBool(matmul, "transpose_x2", false);

    ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("subgraph");
    ge::NodePtr data_a_node = compute_graph_ptr->AddNode(data_a);
    ge::NodePtr data_b_node = compute_graph_ptr->AddNode(data_b);
    ge::NodePtr trans_a_node = compute_graph_ptr->AddNode(trans_a);
    ge::NodePtr trans_b_node = compute_graph_ptr->AddNode(trans_b);
    ge::NodePtr matmul_node = compute_graph_ptr->AddNode(matmul);
    ge::NodePtr trans_c_node = compute_graph_ptr->AddNode(trans_c);
    ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

    ge::GraphUtils::AddEdge(data_a_node->GetOutDataAnchor(0), trans_a_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_b_node->GetOutDataAnchor(0), trans_b_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_a_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_b_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), trans_c_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_c_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

    fe::FusionPassTestUtils::RunGraphFusionPass("AAMatMulNzToNdFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool res = true;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
      if (node->GetType() == "TransData") {
        res = false;
      }
    }
    EXPECT_EQ(res, true);
}

TEST_F(a_a_matmul_nz_to_nd_fusion_pass_test, batchmatmul_v2_without_bias) {
    ge::Graph graph("batchmatmul_v2_without_bias");

    DESC_DATA(data_a, ge::GeShape({16, 512, 1024}), FORMAT_ND, ge::GeShape({16, 512, 1024}), FORMAT_ND, DT_FLOAT16);
    DESC_DATA(data_b, ge::GeShape({16, 1024, 4096}), FORMAT_ND, ge::GeShape({16, 512, 1024}), FORMAT_ND, DT_FLOAT16);

    DESC_DATA(trans_a, ge::GeShape({16, 512, 1024}), FORMAT_ND, ge::GeShape({16, 64, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    DESC_DATA(trans_b, ge::GeShape({16, 1024, 4096}), FORMAT_ND, ge::GeShape({16, 256, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);

    DESC_DATA(batchmatmul_y, ge::GeShape({16, 512, 4096}), FORMAT_ND, ge::GeShape({16, 256, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    DESC_DATA(trans_c, ge::GeShape({16, 512, 1024}), FORMAT_ND, ge::GeShape({16, 512, 1024}), FORMAT_ND, DT_FLOAT16);

    ge::OpDescPtr data_a = std::make_shared<ge::OpDesc>("data_a", "Data");
    ge::OpDescPtr data_b = std::make_shared<ge::OpDesc>("data_b", "Data");
    ge::OpDescPtr trans_a = std::make_shared<ge::OpDesc>("trans_a", "TransData");
    ge::OpDescPtr trans_b = std::make_shared<ge::OpDesc>("trans_b", "TransData");
    ge::OpDescPtr batchmatmul = std::make_shared<ge::OpDesc>("batchmatmul", "BatchMatMulV2");
    ge::OpDescPtr trans_c = std::make_shared<ge::OpDesc>("trans_c", "TransData");
    ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

    data_a->AddOutputDesc(desc_data_a);
    data_b->AddOutputDesc(desc_data_b);
    trans_a->AddInputDesc(desc_data_a);
    trans_a->AddOutputDesc(desc_trans_a);
    trans_b->AddInputDesc(desc_data_b);
    trans_b->AddOutputDesc(desc_trans_b);
    batchmatmul->AddInputDesc("x1", desc_trans_a);
    batchmatmul->AddInputDesc("x2", desc_trans_b);
    batchmatmul->AddOutputDesc(desc_batchmatmul_y);
    trans_c->AddInputDesc(desc_batchmatmul_y);
    trans_c->AddOutputDesc(desc_trans_c);
    netoutput->AddInputDesc(desc_trans_c);

    ge::AttrUtils::SetStr(trans_a, "src_format", "ND");
    ge::AttrUtils::SetStr(trans_a, "dst_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_b, "src_format", "ND");
    ge::AttrUtils::SetStr(trans_b, "dst_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_c, "src_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_c, "dst_format", "ND");
    ge::AttrUtils::SetBool(batchmatmul, "adj_x1", false);
    ge::AttrUtils::SetBool(batchmatmul, "adj_x2", false);

    ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("subgraph");
    ge::NodePtr data_a_node = compute_graph_ptr->AddNode(data_a);
    ge::NodePtr data_b_node = compute_graph_ptr->AddNode(data_b);
    ge::NodePtr trans_a_node = compute_graph_ptr->AddNode(trans_a);
    ge::NodePtr trans_b_node = compute_graph_ptr->AddNode(trans_b);
    ge::NodePtr batchmatmul_node = compute_graph_ptr->AddNode(batchmatmul);
    ge::NodePtr trans_c_node = compute_graph_ptr->AddNode(trans_c);
    ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

    ge::GraphUtils::AddEdge(data_a_node->GetOutDataAnchor(0), trans_a_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_b_node->GetOutDataAnchor(0), trans_b_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_a_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_b_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(batchmatmul_node->GetOutDataAnchor(0), trans_c_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_c_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

    fe::FusionPassTestUtils::RunGraphFusionPass("AAMatMulNzToNdFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool res = true;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
      if (node->GetType() == "TransData") {
        res = false;
      }
    }
    EXPECT_EQ(res, true);
}

TEST_F(a_a_matmul_nz_to_nd_fusion_pass_test, matmul_v2_without_bias) {
    ge::Graph graph("matmul_v2_without_bias");

    DESC_DATA(data_a, ge::GeShape({512, 1024}), FORMAT_ND, ge::GeShape({512, 1024}), FORMAT_ND, DT_FLOAT16);
    DESC_DATA(data_b, ge::GeShape({1024, 4096}), FORMAT_ND, ge::GeShape({1024, 4096}), FORMAT_ND, DT_FLOAT16);

    DESC_DATA(trans_a, ge::GeShape({512, 1024}), FORMAT_ND, ge::GeShape({64, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    DESC_DATA(trans_b, ge::GeShape({1024, 4096}), FORMAT_ND, ge::GeShape({256, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);

    DESC_DATA(matmul_y, ge::GeShape({512, 4096}), FORMAT_ND, ge::GeShape({256, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    DESC_DATA(trans_c, ge::GeShape({512, 4096}), FORMAT_ND, ge::GeShape({512, 4096}), FORMAT_ND, DT_FLOAT16);

    ge::OpDescPtr data_a = std::make_shared<ge::OpDesc>("data_a", "Data");
    ge::OpDescPtr data_b = std::make_shared<ge::OpDesc>("data_b", "Data");
    ge::OpDescPtr trans_a = std::make_shared<ge::OpDesc>("trans_a", "TransData");
    ge::OpDescPtr trans_b = std::make_shared<ge::OpDesc>("trans_b", "TransData");
    ge::OpDescPtr matmul = std::make_shared<ge::OpDesc>("matmul", "MatMulV2");
    ge::OpDescPtr trans_c = std::make_shared<ge::OpDesc>("trans_c", "TransData");
    ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

    data_a->AddOutputDesc(desc_data_a);
    data_b->AddOutputDesc(desc_data_b);
    trans_a->AddInputDesc(desc_data_a);
    trans_a->AddOutputDesc(desc_trans_a);
    trans_b->AddInputDesc(desc_data_b);
    trans_b->AddOutputDesc(desc_trans_b);
    matmul->AddInputDesc("x1", desc_trans_a);
    matmul->AddInputDesc("x2", desc_trans_b);
    matmul->AddOutputDesc(desc_matmul_y);
    trans_c->AddInputDesc(desc_matmul_y);
    trans_c->AddOutputDesc(desc_trans_c);
    netoutput->AddInputDesc(desc_trans_c);

    ge::AttrUtils::SetStr(trans_a, "src_format", "ND");
    ge::AttrUtils::SetStr(trans_a, "dst_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_b, "src_format", "ND");
    ge::AttrUtils::SetStr(trans_b, "dst_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_c, "src_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_c, "dst_format", "ND");
    ge::AttrUtils::SetBool(matmul, "transpose_x1", false);
    ge::AttrUtils::SetBool(matmul, "transpose_x2", false);

    ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("subgraph");
    ge::NodePtr data_a_node = compute_graph_ptr->AddNode(data_a);
    ge::NodePtr data_b_node = compute_graph_ptr->AddNode(data_b);
    ge::NodePtr trans_a_node = compute_graph_ptr->AddNode(trans_a);
    ge::NodePtr trans_b_node = compute_graph_ptr->AddNode(trans_b);
    ge::NodePtr matmul_node = compute_graph_ptr->AddNode(matmul);
    ge::NodePtr trans_c_node = compute_graph_ptr->AddNode(trans_c);
    ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

    ge::GraphUtils::AddEdge(data_a_node->GetOutDataAnchor(0), trans_a_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_b_node->GetOutDataAnchor(0), trans_b_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_a_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_b_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), trans_c_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_c_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

    fe::FusionPassTestUtils::RunGraphFusionPass("AAMatMulNzToNdFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool res = true;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
      if (node->GetType() == "TransData") {
        res = false;
      }
    }
    EXPECT_EQ(res, true);
}