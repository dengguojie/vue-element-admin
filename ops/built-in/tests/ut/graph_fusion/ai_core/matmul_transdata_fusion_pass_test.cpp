#include "gtest/gtest.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "matrix_calculation_ops.h"
#include "transformation_ops.h"
#include "framework/common/types.h"
#define private public
#define protected public
#include "fusion_pass_test_utils.h"
#include "fusion_pass_test_slice_utils.h"
#include "common/util/platform_info.h"


using namespace ge;
using namespace op;

class gemm_transdata_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "gemm_transdata_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "gemm_transdata_fusion_pass_test TearDown" << std::endl;
    }
};

#define DESC_DATA(name, shape_in, format_in, shape_out, format_out, dtype) \
    ge::GeTensorDesc desc_##name(shape_out, format_out, dtype); \
    desc_##name.SetOriginFormat(format_in);                \
    desc_##name.SetOriginShape(shape_in)

void SetPlatForm() {
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_fix_pipe_l0c2out"] = {"float16"};
    opti_compilation_info.soc_version = "soc_version";
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
}

TEST_F(gemm_transdata_fusion_pass_test, gemm_transdata_fusion_pass_test_1) {
    ge::Graph graph("gemm_to_matmul_fusion_fusion_pass_test_1");

    DESC_DATA(data_a, ge::GeShape({16, 32}), FORMAT_ND, ge::GeShape({16, 32}), FORMAT_ND, DT_FLOAT16);
    DESC_DATA(data_b, ge::GeShape({32, 64}), FORMAT_ND, ge::GeShape({32, 64}), FORMAT_ND, DT_FLOAT16);
    DESC_DATA(data_c, ge::GeShape({16, 64}), FORMAT_ND, ge::GeShape({16, 64}), FORMAT_ND, DT_FLOAT16);
    DESC_DATA(data_alpha, ge::GeShape({1}), FORMAT_ND, ge::GeShape({1}), FORMAT_ND, DT_FLOAT16);
    DESC_DATA(data_beta, ge::GeShape({1}), FORMAT_ND, ge::GeShape({1}), FORMAT_ND, DT_FLOAT16);
    DESC_DATA(data_y, ge::GeShape({16, 64}), FORMAT_ND, ge::GeShape({16, 64}), FORMAT_ND, DT_FLOAT16);

    DESC_DATA(data_a_nz, ge::GeShape({16, 32}), FORMAT_ND, ge::GeShape({2, 1, 16,  16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    DESC_DATA(data_b_nz, ge::GeShape({32, 64}), FORMAT_ND, ge::GeShape({4, 2, 16,  16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    DESC_DATA(data_c_nz, ge::GeShape({16, 64}), FORMAT_ND, ge::GeShape({4, 1, 16,  16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    DESC_DATA(data_y_nz, ge::GeShape({16, 64}), FORMAT_ND, ge::GeShape({4, 1, 16,  16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);

    ge::OpDescPtr data_a = std::make_shared<ge::OpDesc>("data_a", "Data");
    ge::OpDescPtr data_b = std::make_shared<ge::OpDesc>("data_b", "Data");
    ge::OpDescPtr data_c = std::make_shared<ge::OpDesc>("data_c", "Data");
    ge::OpDescPtr data_alpha = std::make_shared<ge::OpDesc>("data_alpha", "Data");
    ge::OpDescPtr data_beta = std::make_shared<ge::OpDesc>("data_beta", "Data");
    ge::OpDescPtr matmul = std::make_shared<ge::OpDesc>("matmul", "MatMulV2");
    ge::OpDescPtr mul_1 = std::make_shared<ge::OpDesc>("mul_1", "Mul");
    ge::OpDescPtr mul_2 = std::make_shared<ge::OpDesc>("mul_2", "Mul");
    ge::OpDescPtr add = std::make_shared<ge::OpDesc>("add", "Add");
    ge::OpDescPtr trans_a = std::make_shared<ge::OpDesc>("trans_a", "TransData");
    ge::OpDescPtr trans_b = std::make_shared<ge::OpDesc>("trans_b", "TransData");
    ge::OpDescPtr trans_c = std::make_shared<ge::OpDesc>("trans_c", "TransData");
    ge::OpDescPtr trans_y = std::make_shared<ge::OpDesc>("trans_y", "TransData");
    ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

    // update input data desc
    data_a->AddOutputDesc(desc_data_a);
    data_b->AddOutputDesc(desc_data_b);
    data_c->AddOutputDesc(desc_data_c);
    data_alpha->AddOutputDesc(desc_data_alpha);
    data_beta->AddOutputDesc(desc_data_beta);
    trans_a->AddInputDesc(desc_data_a);
    trans_b->AddInputDesc(desc_data_b);
    trans_c->AddInputDesc(desc_data_c);
    trans_a->AddOutputDesc(desc_data_a_nz);
    trans_b->AddOutputDesc(desc_data_b_nz);
    trans_c->AddOutputDesc(desc_data_c_nz);
    // update matmul desc
    matmul->AddInputDesc("x1", desc_data_a_nz);
    matmul->AddInputDesc("x2", desc_data_b_nz);
    matmul->AddOutputDesc(desc_data_y_nz);
    // update mul and add
    mul_1->AddInputDesc("x1", desc_data_y_nz);
    mul_1->AddInputDesc("x2", desc_data_alpha);
    mul_1->AddOutputDesc(desc_data_y_nz);
    mul_2->AddInputDesc("x1", desc_data_c_nz);
    mul_2->AddInputDesc("x2", desc_data_beta);
    mul_2->AddOutputDesc(desc_data_c_nz);
    add->AddInputDesc("x1", desc_data_y_nz);
    add->AddInputDesc("x2", desc_data_c_nz);
    add->AddOutputDesc(desc_data_y_nz);
    // update out
    trans_y->AddInputDesc(desc_data_y_nz);
    trans_y->AddOutputDesc(desc_data_y);
    netoutput->AddInputDesc(desc_data_y);
    // set attr
    ge::AttrUtils::SetBool(matmul, "transpose_x1", false);
    ge::AttrUtils::SetBool(matmul, "transpose_x2", false);
    ge::AttrUtils::SetBool(trans_a, "src_format", "ND");
    ge::AttrUtils::SetBool(trans_a, "dst_format", "FRACTAL_NZ");
    ge::AttrUtils::SetBool(trans_b, "src_format", "ND");
    ge::AttrUtils::SetBool(trans_b, "dst_format", "FRACTAL_NZ");
    ge::AttrUtils::SetBool(trans_c, "src_format", "ND");
    ge::AttrUtils::SetBool(trans_c, "dst_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_y, "src_format", "FRACTAL_NZ");
    ge::AttrUtils::SetStr(trans_y, "dst_format", "ND");

    ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("subgraph");
    ge::NodePtr data_a_node = compute_graph_ptr->AddNode(data_a);
    ge::NodePtr data_b_node = compute_graph_ptr->AddNode(data_b);
    ge::NodePtr data_c_node = compute_graph_ptr->AddNode(data_c);
    ge::NodePtr data_alpha_node = compute_graph_ptr->AddNode(data_alpha);
    ge::NodePtr data_beta_node = compute_graph_ptr->AddNode(data_beta);
    ge::NodePtr matmul_node = compute_graph_ptr->AddNode(matmul);
    ge::NodePtr mul_1_node = compute_graph_ptr->AddNode(mul_1);
    ge::NodePtr mul_2_node = compute_graph_ptr->AddNode(mul_2);
    ge::NodePtr add_node = compute_graph_ptr->AddNode(add);
    ge::NodePtr trans_a_node = compute_graph_ptr->AddNode(trans_a);
    ge::NodePtr trans_b_node = compute_graph_ptr->AddNode(trans_b);
    ge::NodePtr trans_c_node = compute_graph_ptr->AddNode(trans_c);
    ge::NodePtr trans_y_node = compute_graph_ptr->AddNode(trans_y);
    ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

    ge::GraphUtils::AddEdge(data_a_node->GetOutDataAnchor(0), trans_a_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_b_node->GetOutDataAnchor(0), trans_b_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_c_node->GetOutDataAnchor(0), trans_c_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_a_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_b_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), mul_1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_alpha_node->GetOutDataAnchor(0), mul_1_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(trans_c_node->GetOutDataAnchor(0), mul_2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_beta_node->GetOutDataAnchor(0), mul_2_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(mul_1_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(mul_2_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), trans_y_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(trans_y_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

    SetPlatForm();
    fe::FusionPassTestUtils::RunGraphFusionPass("MatMulTransdataFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    int trans_num = 0;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
      if (node->GetType() == "TransData") {
        trans_num++;
      }
    }
    EXPECT_EQ(trans_num, 3);
}
