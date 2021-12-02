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

class gemm_to_matmul_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "gemm_to_matmul_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "gemm_to_matmul_fusion_pass_test TearDown" << std::endl;
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


TEST_F(gemm_to_matmul_fusion_pass_test, gemm_to_matmul_fusion_pass_test_1) {
    ge::Graph graph("gemm_to_matmul_fusion_fusion_pass_test_1");

    DESC_DATA(data_a, ge::GeShape({16, 32}), FORMAT_ND, ge::GeShape({16, 32}), FORMAT_ND, DT_FLOAT16);
    DESC_DATA(data_b, ge::GeShape({32, 64}), FORMAT_ND, ge::GeShape({32, 64}), FORMAT_ND, DT_FLOAT16);
    DESC_DATA(data_c, ge::GeShape({16, 64}), FORMAT_ND, ge::GeShape({16, 64}), FORMAT_ND, DT_FLOAT16);
    DESC_DATA(data_alpha, ge::GeShape({1}), FORMAT_ND, ge::GeShape({1}), FORMAT_ND, DT_FLOAT16);
    DESC_DATA(data_beta, ge::GeShape({1}), FORMAT_ND, ge::GeShape({1}), FORMAT_ND, DT_FLOAT16);
    DESC_DATA(data_y, ge::GeShape({16, 64}), FORMAT_ND, ge::GeShape({16, 64}), FORMAT_ND, DT_FLOAT16);

    ge::OpDescPtr data_a = std::make_shared<ge::OpDesc>("data_a", "Data");
    ge::OpDescPtr data_b = std::make_shared<ge::OpDesc>("data_b", "Data");
    ge::OpDescPtr data_c = std::make_shared<ge::OpDesc>("data_c", "Data");
    ge::OpDescPtr data_alpha = std::make_shared<ge::OpDesc>("data_alpha", "Data");
    ge::OpDescPtr data_beta = std::make_shared<ge::OpDesc>("data_beta", "Data");
    ge::OpDescPtr gemm = std::make_shared<ge::OpDesc>("gemm", "GEMM");
    ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

    // updata input data desc
    data_a->AddOutputDesc(desc_data_a);
    data_b->AddOutputDesc(desc_data_b);
    data_c->AddOutputDesc(desc_data_c);
    data_alpha->AddOutputDesc(desc_data_alpha);
    data_beta->AddOutputDesc(desc_data_beta);
    // updata gemm desc
    gemm->AddInputDesc("a", desc_data_a);
    gemm->AddInputDesc("b", desc_data_b);
    gemm->AddInputDesc("c", desc_data_c);
    gemm->AddInputDesc("alpha", desc_data_alpha);
    gemm->AddInputDesc("beta", desc_data_beta);
    gemm->AddOutputDesc(desc_data_y);
    netoutput->AddInputDesc(desc_data_y);
    // set attr
    ge::AttrUtils::SetBool(gemm, "transpose_a", false);
    ge::AttrUtils::SetBool(gemm, "transpose_b", false);

    ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("subgraph");
    ge::NodePtr data_a_node = compute_graph_ptr->AddNode(data_a);
    ge::NodePtr data_b_node = compute_graph_ptr->AddNode(data_b);
    ge::NodePtr data_c_node = compute_graph_ptr->AddNode(data_c);
    ge::NodePtr data_alpha_node = compute_graph_ptr->AddNode(data_alpha);
    ge::NodePtr data_beta_node = compute_graph_ptr->AddNode(data_beta);
    ge::NodePtr gemm_node = compute_graph_ptr->AddNode(gemm);
    ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

    ge::GraphUtils::AddEdge(data_a_node->GetOutDataAnchor(0), gemm_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_b_node->GetOutDataAnchor(0), gemm_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(data_c_node->GetOutDataAnchor(0), gemm_node->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(data_alpha_node->GetOutDataAnchor(0), gemm_node->GetInDataAnchor(3));
    ge::GraphUtils::AddEdge(data_beta_node->GetOutDataAnchor(0), gemm_node->GetInDataAnchor(4));
    ge::GraphUtils::AddEdge(gemm_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

    SetPlatForm();
    fe::FusionPassTestUtils::RunGraphFusionPass("GemmToMatmulFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool find_matmul = false;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
      if (node->GetType() == "MatMulV2") {
        find_matmul = true;
      }
    }
    EXPECT_EQ(find_matmul, true);
}
