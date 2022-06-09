#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "matrix_calculation_ops.h"
#include "nn_norm_ops.h"
#include "transformation_ops.h"
#include "nonlinear_fuc_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/operator_reg.h"
#include "fc_transdata_merge_fusion_pass_test.h"
#define private public
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;

class attention_qkv_gradx_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "attention_qkv_gradx_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "attention_qkv_gradx_fusion_test TearDown" << std::endl;
  }
};

TEST_F(attention_qkv_gradx_fusion_test, attention_qkv_gradx_fusion_test_1) {
    // success testcase
    ge::Graph graph("attention_qkv_gradx_fusion_test1");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // ln_bp
    auto ln_dy_input_data = op::Data("ln_dy_input_data");
    TensorDesc ln_dy_input_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    ln_dy_input_data.update_input_desc_x(ln_dy_input_desc);
    ln_dy_input_data.update_output_desc_y(ln_dy_input_desc);

    auto ln_x_input_data = op::Data("ln_x_input_data");
    TensorDesc ln_x_input_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    ln_x_input_data.update_input_desc_x(ln_x_input_desc);
    ln_x_input_data.update_output_desc_y(ln_x_input_desc);

    auto ln_mean_input_data = op::Data("ln_mean_input_data");
    TensorDesc ln_mean_input_desc(ge::Shape({12288}), FORMAT_NHWC, DT_FLOAT16);
    ln_mean_input_data.update_input_desc_x(ln_mean_input_desc);
    ln_mean_input_data.update_output_desc_y(ln_mean_input_desc);

    auto ln_variance_data = op::Data("ln_variance_data");
    TensorDesc ln_variance_input_desc(ge::Shape({12288}), FORMAT_NHWC, DT_FLOAT16);
    ln_variance_data.update_input_desc_x(ln_variance_input_desc);
    ln_variance_data.update_output_desc_y(ln_variance_input_desc);

    auto ln_gamma_data = op::Data("ln_gamma_data");
    TensorDesc ln_gamma_input_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    ln_gamma_data.update_input_desc_x(ln_gamma_input_desc);
    ln_gamma_data.update_output_desc_y(ln_gamma_input_desc);

    auto ln_bp = op::LayerNormXBackpropV2("ln_bp");
    ln_bp.set_input_dy(ln_dy_input_data);
    ln_bp.set_input_x(ln_x_input_data);
    ln_bp.set_input_mean(ln_mean_input_data);
    ln_bp.set_input_variance(ln_variance_data);
    ln_bp.set_input_gamma(ln_gamma_data);

    // matmul_dx_qkv
    auto query_x_data = op::Data("query_x_data");
    TensorDesc query_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    query_x_data.update_input_desc_x(query_x_desc);
    query_x_data.update_output_desc_y(query_x_desc);

    auto key_x_data = op::Data("key_x_data");
    TensorDesc key_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    key_x_data.update_input_desc_x(key_x_desc);
    key_x_data.update_output_desc_y(key_x_desc);

    auto value_x_data = op::Data("value_x_data");
    TensorDesc value_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    value_x_data.update_input_desc_x(value_x_desc);
    value_x_data.update_output_desc_y(value_x_desc);

    auto kernel_query_data = op::Data("kernel_query_data");
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_query_data.update_input_desc_x(kernel_query_desc);
    kernel_query_data.update_output_desc_y(kernel_query_desc);

    auto kernel_key_data = op::Data("kernel_key_data");
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_key_data.update_input_desc_x(kernel_key_desc);
    kernel_key_data.update_output_desc_y(kernel_key_desc);

    auto kernel_value_data = op::Data("kernel_value_data");
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_value_data.update_input_desc_x(kernel_value_desc);
    kernel_value_data.update_output_desc_y(kernel_value_desc);

    auto matmul_query = op::MatMulV2("matmul_query");
    matmul_query.set_input_x1(query_x_data);
    matmul_query.set_input_x2(kernel_query_data);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(true);

    auto matmul_key = op::MatMulV2("matmul_key");
    matmul_key.set_input_x1(key_x_data);
    matmul_key.set_input_x2(kernel_key_data);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(true);

    auto matmul_value = op::MatMulV2("matmul_value");
    matmul_value.set_input_x1(value_x_data);
    matmul_value.set_input_x2(kernel_value_data);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(true);

    auto addn = op::AddN("addn");
    addn.create_dynamic_input_x(4);
    addn.set_dynamic_input_x(0, ln_bp, "pd_x");
    addn.set_dynamic_input_x(1, matmul_query);
    addn.set_dynamic_input_x(2, matmul_key);
    addn.set_dynamic_input_x(3, matmul_value);
    addn.set_attr_N(4);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    addn.update_output_desc_y(output_desc_y);

    std::vector<Operator> inputs{query_x_data};
    std::vector<Operator> outputs{addn};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 768, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradX") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, true);
    EXPECT_EQ(shapeMatch, true);
}

TEST_F(attention_qkv_gradx_fusion_test, attention_qkv_gradx_fusion_test_2) {
    // addn input num less than 4
    ge::Graph graph("attention_qkv_gradx_fusion_test2");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // matmul_dx_qkv
    auto query_x_data = op::Data("query_x_data");
    TensorDesc query_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    query_x_data.update_input_desc_x(query_x_desc);
    query_x_data.update_output_desc_y(query_x_desc);

    auto key_x_data = op::Data("key_x_data");
    TensorDesc key_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    key_x_data.update_input_desc_x(key_x_desc);
    key_x_data.update_output_desc_y(key_x_desc);

    auto value_x_data = op::Data("value_x_data");
    TensorDesc value_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    value_x_data.update_input_desc_x(value_x_desc);
    value_x_data.update_output_desc_y(value_x_desc);

    auto kernel_query_data = op::Data("kernel_query_data");
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_query_data.update_input_desc_x(kernel_query_desc);
    kernel_query_data.update_output_desc_y(kernel_query_desc);

    auto kernel_key_data = op::Data("kernel_key_data");
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_key_data.update_input_desc_x(kernel_key_desc);
    kernel_key_data.update_output_desc_y(kernel_key_desc);

    auto kernel_value_data = op::Data("kernel_value_data");
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_value_data.update_input_desc_x(kernel_value_desc);
    kernel_value_data.update_output_desc_y(kernel_value_desc);

    auto matmul_query = op::MatMulV2("matmul_query");
    matmul_query.set_input_x1(query_x_data);
    matmul_query.set_input_x2(kernel_query_data);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(true);

    auto matmul_key = op::MatMulV2("matmul_key");
    matmul_key.set_input_x1(key_x_data);
    matmul_key.set_input_x2(kernel_key_data);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(true);

    auto matmul_value = op::MatMulV2("matmul_value");
    matmul_value.set_input_x1(value_x_data);
    matmul_value.set_input_x2(kernel_value_data);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(true);

    auto addn = op::AddN("addn");
    addn.create_dynamic_input_x(3);
    addn.set_dynamic_input_x(0, matmul_query);
    addn.set_dynamic_input_x(1, matmul_key);
    addn.set_dynamic_input_x(2, matmul_value);
    addn.set_attr_N(3);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    addn.update_output_desc_y(output_desc_y);

    std::vector<Operator> inputs{query_x_data};
    std::vector<Operator> outputs{addn};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 768, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradX") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}

TEST_F(attention_qkv_gradx_fusion_test, attention_qkv_gradx_fusion_test_3) {
    // matmulv2->addn pattern match failed
    ge::Graph graph("attention_qkv_gradx_fusion_test3");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // ln_bp
    auto ln_dy_input_data = op::Data("ln_dy_input_data");
    TensorDesc ln_dy_input_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    ln_dy_input_data.update_input_desc_x(ln_dy_input_desc);
    ln_dy_input_data.update_output_desc_y(ln_dy_input_desc);

    auto ln_x_input_data = op::Data("ln_x_input_data");
    TensorDesc ln_x_input_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    ln_x_input_data.update_input_desc_x(ln_x_input_desc);
    ln_x_input_data.update_output_desc_y(ln_x_input_desc);

    auto ln_mean_input_data = op::Data("ln_mean_input_data");
    TensorDesc ln_mean_input_desc(ge::Shape({12288}), FORMAT_NHWC, DT_FLOAT16);
    ln_mean_input_data.update_input_desc_x(ln_mean_input_desc);
    ln_mean_input_data.update_output_desc_y(ln_mean_input_desc);

    auto ln_variance_data = op::Data("ln_variance_data");
    TensorDesc ln_variance_input_desc(ge::Shape({12288}), FORMAT_NHWC, DT_FLOAT16);
    ln_variance_data.update_input_desc_x(ln_variance_input_desc);
    ln_variance_data.update_output_desc_y(ln_variance_input_desc);

    auto ln_gamma_data = op::Data("ln_gamma_data");
    TensorDesc ln_gamma_input_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    ln_gamma_data.update_input_desc_x(ln_gamma_input_desc);
    ln_gamma_data.update_output_desc_y(ln_gamma_input_desc);

    auto ln_bp = op::LayerNormXBackpropV2("ln_bp");
    ln_bp.set_input_dy(ln_dy_input_data);
    ln_bp.set_input_x(ln_x_input_data);
    ln_bp.set_input_mean(ln_mean_input_data);
    ln_bp.set_input_variance(ln_variance_data);
    ln_bp.set_input_gamma(ln_gamma_data);

    // matmul_dx_qkv
    auto query_x_data = op::Data("query_x_data");
    TensorDesc query_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    query_x_data.update_input_desc_x(query_x_desc);
    query_x_data.update_output_desc_y(query_x_desc);

    auto key_x_data = op::Data("key_x_data");
    TensorDesc key_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    key_x_data.update_input_desc_x(key_x_desc);
    key_x_data.update_output_desc_y(key_x_desc);

    auto value_x_data = op::Data("value_x_data");
    TensorDesc value_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    value_x_data.update_input_desc_x(value_x_desc);
    value_x_data.update_output_desc_y(value_x_desc);

    auto kernel_query_data = op::Data("kernel_query_data");
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_query_data.update_input_desc_x(kernel_query_desc);
    kernel_query_data.update_output_desc_y(kernel_query_desc);

    auto kernel_key_data = op::Data("kernel_key_data");
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_key_data.update_input_desc_x(kernel_key_desc);
    kernel_key_data.update_output_desc_y(kernel_key_desc);

    auto kernel_value_data = op::Data("kernel_value_data");
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_value_data.update_input_desc_x(kernel_value_desc);
    kernel_value_data.update_output_desc_y(kernel_value_desc);

    auto matmul_query = op::MatMul("matmul_query");
    matmul_query.set_input_x1(query_x_data);
    matmul_query.set_input_x2(kernel_query_data);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(true);

    auto matmul_key = op::MatMulV2("matmul_key");
    matmul_key.set_input_x1(key_x_data);
    matmul_key.set_input_x2(kernel_key_data);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(true);

    auto matmul_value = op::MatMulV2("matmul_value");
    matmul_value.set_input_x1(value_x_data);
    matmul_value.set_input_x2(kernel_value_data);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(true);

    auto addn = op::AddN("addn");
    addn.create_dynamic_input_x(4);
    addn.set_dynamic_input_x(0, ln_bp, "pd_x");
    addn.set_dynamic_input_x(1, matmul_query);
    addn.set_dynamic_input_x(2, matmul_key);
    addn.set_dynamic_input_x(3, matmul_value);
    addn.set_attr_N(4);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    addn.update_output_desc_y(output_desc_y);

    std::vector<Operator> inputs{query_x_data};
    std::vector<Operator> outputs{addn};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 768, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradX") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}

TEST_F(attention_qkv_gradx_fusion_test, attention_qkv_gradx_fusion_test_4) {
    // transpose of matmul match failed
    ge::Graph graph("attention_qkv_gradx_fusion_test4");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // ln_bp
    auto ln_dy_input_data = op::Data("ln_dy_input_data");
    TensorDesc ln_dy_input_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    ln_dy_input_data.update_input_desc_x(ln_dy_input_desc);
    ln_dy_input_data.update_output_desc_y(ln_dy_input_desc);

    auto ln_x_input_data = op::Data("ln_x_input_data");
    TensorDesc ln_x_input_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    ln_x_input_data.update_input_desc_x(ln_x_input_desc);
    ln_x_input_data.update_output_desc_y(ln_x_input_desc);

    auto ln_mean_input_data = op::Data("ln_mean_input_data");
    TensorDesc ln_mean_input_desc(ge::Shape({12288}), FORMAT_NHWC, DT_FLOAT16);
    ln_mean_input_data.update_input_desc_x(ln_mean_input_desc);
    ln_mean_input_data.update_output_desc_y(ln_mean_input_desc);

    auto ln_variance_data = op::Data("ln_variance_data");
    TensorDesc ln_variance_input_desc(ge::Shape({12288}), FORMAT_NHWC, DT_FLOAT16);
    ln_variance_data.update_input_desc_x(ln_variance_input_desc);
    ln_variance_data.update_output_desc_y(ln_variance_input_desc);

    auto ln_gamma_data = op::Data("ln_gamma_data");
    TensorDesc ln_gamma_input_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    ln_gamma_data.update_input_desc_x(ln_gamma_input_desc);
    ln_gamma_data.update_output_desc_y(ln_gamma_input_desc);

    auto ln_bp = op::LayerNormXBackpropV2("ln_bp");
    ln_bp.set_input_dy(ln_dy_input_data);
    ln_bp.set_input_x(ln_x_input_data);
    ln_bp.set_input_mean(ln_mean_input_data);
    ln_bp.set_input_variance(ln_variance_data);
    ln_bp.set_input_gamma(ln_gamma_data);

    // matmul_dx_qkv
    auto query_x_data = op::Data("query_x_data");
    TensorDesc query_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    query_x_data.update_input_desc_x(query_x_desc);
    query_x_data.update_output_desc_y(query_x_desc);

    auto key_x_data = op::Data("key_x_data");
    TensorDesc key_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    key_x_data.update_input_desc_x(key_x_desc);
    key_x_data.update_output_desc_y(key_x_desc);

    auto value_x_data = op::Data("value_x_data");
    TensorDesc value_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    value_x_data.update_input_desc_x(value_x_desc);
    value_x_data.update_output_desc_y(value_x_desc);

    auto kernel_query_data = op::Data("kernel_query_data");
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_query_data.update_input_desc_x(kernel_query_desc);
    kernel_query_data.update_output_desc_y(kernel_query_desc);

    auto kernel_key_data = op::Data("kernel_key_data");
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_key_data.update_input_desc_x(kernel_key_desc);
    kernel_key_data.update_output_desc_y(kernel_key_desc);

    auto kernel_value_data = op::Data("kernel_value_data");
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_value_data.update_input_desc_x(kernel_value_desc);
    kernel_value_data.update_output_desc_y(kernel_value_desc);

    auto matmul_query = op::MatMulV2("matmul_query");
    matmul_query.set_input_x1(query_x_data);
    matmul_query.set_input_x2(kernel_query_data);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(false);

    auto matmul_key = op::MatMulV2("matmul_key");
    matmul_key.set_input_x1(key_x_data);
    matmul_key.set_input_x2(kernel_key_data);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(false);

    auto matmul_value = op::MatMulV2("matmul_value");
    matmul_value.set_input_x1(value_x_data);
    matmul_value.set_input_x2(kernel_value_data);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(false);

    auto addn = op::AddN("addn");
    addn.create_dynamic_input_x(4);
    addn.set_dynamic_input_x(0, ln_bp, "pd_x");
    addn.set_dynamic_input_x(1, matmul_query);
    addn.set_dynamic_input_x(2, matmul_key);
    addn.set_dynamic_input_x(3, matmul_value);
    addn.set_attr_N(4);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    addn.update_output_desc_y(output_desc_y);

    std::vector<Operator> inputs{query_x_data};
    std::vector<Operator> outputs{addn};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 768, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradX") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}

TEST_F(attention_qkv_gradx_fusion_test, attention_qkv_gradx_fusion_test_5) {
    // invalid addn_out_shape
    ge::Graph graph("attention_qkv_gradx_fusion_test5");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // ln_bp
    auto ln_dy_input_data = op::Data("ln_dy_input_data");
    TensorDesc ln_dy_input_desc(ge::Shape({32, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    ln_dy_input_data.update_input_desc_x(ln_dy_input_desc);
    ln_dy_input_data.update_output_desc_y(ln_dy_input_desc);

    auto ln_x_input_data = op::Data("ln_x_input_data");
    TensorDesc ln_x_input_desc(ge::Shape({32, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    ln_x_input_data.update_input_desc_x(ln_x_input_desc);
    ln_x_input_data.update_output_desc_y(ln_x_input_desc);

    auto ln_mean_input_data = op::Data("ln_mean_input_data");
    TensorDesc ln_mean_input_desc(ge::Shape({12288}), FORMAT_NHWC, DT_FLOAT16);
    ln_mean_input_data.update_input_desc_x(ln_mean_input_desc);
    ln_mean_input_data.update_output_desc_y(ln_mean_input_desc);

    auto ln_variance_data = op::Data("ln_variance_data");
    TensorDesc ln_variance_input_desc(ge::Shape({12288}), FORMAT_NHWC, DT_FLOAT16);
    ln_variance_data.update_input_desc_x(ln_variance_input_desc);
    ln_variance_data.update_output_desc_y(ln_variance_input_desc);

    auto ln_gamma_data = op::Data("ln_gamma_data");
    TensorDesc ln_gamma_input_desc(ge::Shape({512}), FORMAT_NHWC, DT_FLOAT16);
    ln_gamma_data.update_input_desc_x(ln_gamma_input_desc);
    ln_gamma_data.update_output_desc_y(ln_gamma_input_desc);

    auto ln_bp = op::LayerNormXBackpropV2("ln_bp");
    ln_bp.set_input_dy(ln_dy_input_data);
    ln_bp.set_input_x(ln_x_input_data);
    ln_bp.set_input_mean(ln_mean_input_data);
    ln_bp.set_input_variance(ln_variance_data);
    ln_bp.set_input_gamma(ln_gamma_data);

    // matmul_dx_qkv
    auto query_x_data = op::Data("query_x_data");
    TensorDesc query_x_desc(ge::Shape({32, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    query_x_data.update_input_desc_x(query_x_desc);
    query_x_data.update_output_desc_y(query_x_desc);

    auto key_x_data = op::Data("key_x_data");
    TensorDesc key_x_desc(ge::Shape({32, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    key_x_data.update_input_desc_x(key_x_desc);
    key_x_data.update_output_desc_y(key_x_desc);

    auto value_x_data = op::Data("value_x_data");
    TensorDesc value_x_desc(ge::Shape({32, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    value_x_data.update_input_desc_x(value_x_desc);
    value_x_data.update_output_desc_y(value_x_desc);

    auto kernel_query_data = op::Data("kernel_query_data");
    TensorDesc kernel_query_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_query_data.update_input_desc_x(kernel_query_desc);
    kernel_query_data.update_output_desc_y(kernel_query_desc);

    auto kernel_key_data = op::Data("kernel_key_data");
    TensorDesc kernel_key_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_key_data.update_input_desc_x(kernel_key_desc);
    kernel_key_data.update_output_desc_y(kernel_key_desc);

    auto kernel_value_data = op::Data("kernel_value_data");
    TensorDesc kernel_value_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_value_data.update_input_desc_x(kernel_value_desc);
    kernel_value_data.update_output_desc_y(kernel_value_desc);

    auto matmul_query = op::MatMulV2("matmul_query");
    matmul_query.set_input_x1(query_x_data);
    matmul_query.set_input_x2(kernel_query_data);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(true);

    auto matmul_key = op::MatMulV2("matmul_key");
    matmul_key.set_input_x1(key_x_data);
    matmul_key.set_input_x2(kernel_key_data);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(true);

    auto matmul_value = op::MatMulV2("matmul_value");
    matmul_value.set_input_x1(value_x_data);
    matmul_value.set_input_x2(kernel_value_data);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(true);

    auto addn = op::AddN("addn");
    addn.create_dynamic_input_x(4);
    addn.set_dynamic_input_x(0, ln_bp, "pd_x");
    addn.set_dynamic_input_x(1, matmul_query);
    addn.set_dynamic_input_x(2, matmul_key);
    addn.set_dynamic_input_x(3, matmul_value);
    addn.set_attr_N(4);
    TensorDesc output_desc_y(ge::Shape({32, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 512}));
    addn.update_output_desc_y(output_desc_y);

    std::vector<Operator> inputs{query_x_data};
    std::vector<Operator> outputs{addn};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{32, 768, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradX") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}

TEST_F(attention_qkv_gradx_fusion_test, attention_qkv_gradx_fusion_test_6) {
    // platform unsupported
    ge::Graph graph("attention_qkv_gradx_fusion_test6");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 8;
    opti_compilation_info.soc_version = "soc_version";
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // ln_bp
    auto ln_dy_input_data = op::Data("ln_dy_input_data");
    TensorDesc ln_dy_input_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    ln_dy_input_data.update_input_desc_x(ln_dy_input_desc);
    ln_dy_input_data.update_output_desc_y(ln_dy_input_desc);

    auto ln_x_input_data = op::Data("ln_x_input_data");
    TensorDesc ln_x_input_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    ln_x_input_data.update_input_desc_x(ln_x_input_desc);
    ln_x_input_data.update_output_desc_y(ln_x_input_desc);

    auto ln_mean_input_data = op::Data("ln_mean_input_data");
    TensorDesc ln_mean_input_desc(ge::Shape({12288}), FORMAT_NHWC, DT_FLOAT16);
    ln_mean_input_data.update_input_desc_x(ln_mean_input_desc);
    ln_mean_input_data.update_output_desc_y(ln_mean_input_desc);

    auto ln_variance_data = op::Data("ln_variance_data");
    TensorDesc ln_variance_input_desc(ge::Shape({12288}), FORMAT_NHWC, DT_FLOAT16);
    ln_variance_data.update_input_desc_x(ln_variance_input_desc);
    ln_variance_data.update_output_desc_y(ln_variance_input_desc);

    auto ln_gamma_data = op::Data("ln_gamma_data");
    TensorDesc ln_gamma_input_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    ln_gamma_data.update_input_desc_x(ln_gamma_input_desc);
    ln_gamma_data.update_output_desc_y(ln_gamma_input_desc);

    auto ln_bp = op::LayerNormXBackpropV2("ln_bp");
    ln_bp.set_input_dy(ln_dy_input_data);
    ln_bp.set_input_x(ln_x_input_data);
    ln_bp.set_input_mean(ln_mean_input_data);
    ln_bp.set_input_variance(ln_variance_data);
    ln_bp.set_input_gamma(ln_gamma_data);

    // matmul_dx_qkv
    auto query_x_data = op::Data("query_x_data");
    TensorDesc query_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    query_x_data.update_input_desc_x(query_x_desc);
    query_x_data.update_output_desc_y(query_x_desc);

    auto key_x_data = op::Data("key_x_data");
    TensorDesc key_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    key_x_data.update_input_desc_x(key_x_desc);
    key_x_data.update_output_desc_y(key_x_desc);

    auto value_x_data = op::Data("value_x_data");
    TensorDesc value_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    value_x_data.update_input_desc_x(value_x_desc);
    value_x_data.update_output_desc_y(value_x_desc);

    auto kernel_query_data = op::Data("kernel_query_data");
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_query_data.update_input_desc_x(kernel_query_desc);
    kernel_query_data.update_output_desc_y(kernel_query_desc);

    auto kernel_key_data = op::Data("kernel_key_data");
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_key_data.update_input_desc_x(kernel_key_desc);
    kernel_key_data.update_output_desc_y(kernel_key_desc);

    auto kernel_value_data = op::Data("kernel_value_data");
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_value_data.update_input_desc_x(kernel_value_desc);
    kernel_value_data.update_output_desc_y(kernel_value_desc);

    auto matmul_query = op::MatMulV2("matmul_query");
    matmul_query.set_input_x1(query_x_data);
    matmul_query.set_input_x2(kernel_query_data);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(true);

    auto matmul_key = op::MatMulV2("matmul_key");
    matmul_key.set_input_x1(key_x_data);
    matmul_key.set_input_x2(kernel_key_data);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(true);

    auto matmul_value = op::MatMulV2("matmul_value");
    matmul_value.set_input_x1(value_x_data);
    matmul_value.set_input_x2(kernel_value_data);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(true);

    auto addn = op::AddN("addn");
    addn.create_dynamic_input_x(4);
    addn.set_dynamic_input_x(0, ln_bp, "pd_x");
    addn.set_dynamic_input_x(1, matmul_query);
    addn.set_dynamic_input_x(2, matmul_key);
    addn.set_dynamic_input_x(3, matmul_value);
    addn.set_attr_N(4);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    addn.update_output_desc_y(output_desc_y);

    std::vector<Operator> inputs{query_x_data};
    std::vector<Operator> outputs{addn};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 768, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradX") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}

TEST_F(attention_qkv_gradx_fusion_test, attention_qkv_gradx_fusion_test_7) {
    // unsupported dtype
    ge::Graph graph("attention_qkv_gradx_fusion_test7");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // ln_bp
    auto ln_dy_input_data = op::Data("ln_dy_input_data");
    TensorDesc ln_dy_input_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    ln_dy_input_data.update_input_desc_x(ln_dy_input_desc);
    ln_dy_input_data.update_output_desc_y(ln_dy_input_desc);

    auto ln_x_input_data = op::Data("ln_x_input_data");
    TensorDesc ln_x_input_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    ln_x_input_data.update_input_desc_x(ln_x_input_desc);
    ln_x_input_data.update_output_desc_y(ln_x_input_desc);

    auto ln_mean_input_data = op::Data("ln_mean_input_data");
    TensorDesc ln_mean_input_desc(ge::Shape({12288}), FORMAT_NHWC, DT_FLOAT16);
    ln_mean_input_data.update_input_desc_x(ln_mean_input_desc);
    ln_mean_input_data.update_output_desc_y(ln_mean_input_desc);

    auto ln_variance_data = op::Data("ln_variance_data");
    TensorDesc ln_variance_input_desc(ge::Shape({12288}), FORMAT_NHWC, DT_FLOAT16);
    ln_variance_data.update_input_desc_x(ln_variance_input_desc);
    ln_variance_data.update_output_desc_y(ln_variance_input_desc);

    auto ln_gamma_data = op::Data("ln_gamma_data");
    TensorDesc ln_gamma_input_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    ln_gamma_data.update_input_desc_x(ln_gamma_input_desc);
    ln_gamma_data.update_output_desc_y(ln_gamma_input_desc);

    auto ln_bp = op::LayerNormXBackpropV2("ln_bp");
    ln_bp.set_input_dy(ln_dy_input_data);
    ln_bp.set_input_x(ln_x_input_data);
    ln_bp.set_input_mean(ln_mean_input_data);
    ln_bp.set_input_variance(ln_variance_data);
    ln_bp.set_input_gamma(ln_gamma_data);

    // matmul_dx_qkv
    auto query_x_data = op::Data("query_x_data");
    TensorDesc query_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    query_x_data.update_input_desc_x(query_x_desc);
    query_x_data.update_output_desc_y(query_x_desc);

    auto key_x_data = op::Data("key_x_data");
    TensorDesc key_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    key_x_data.update_input_desc_x(key_x_desc);
    key_x_data.update_output_desc_y(key_x_desc);

    auto value_x_data = op::Data("value_x_data");
    TensorDesc value_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    value_x_data.update_input_desc_x(value_x_desc);
    value_x_data.update_output_desc_y(value_x_desc);

    auto kernel_query_data = op::Data("kernel_query_data");
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_query_data.update_input_desc_x(kernel_query_desc);
    kernel_query_data.update_output_desc_y(kernel_query_desc);

    auto kernel_key_data = op::Data("kernel_key_data");
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_key_data.update_input_desc_x(kernel_key_desc);
    kernel_key_data.update_output_desc_y(kernel_key_desc);

    auto kernel_value_data = op::Data("kernel_value_data");
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_value_data.update_input_desc_x(kernel_value_desc);
    kernel_value_data.update_output_desc_y(kernel_value_desc);

    auto matmul_query = op::MatMulV2("matmul_query");
    matmul_query.set_input_x1(query_x_data);
    matmul_query.set_input_x2(kernel_query_data);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(true);

    auto matmul_key = op::MatMulV2("matmul_key");
    matmul_key.set_input_x1(key_x_data);
    matmul_key.set_input_x2(kernel_key_data);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(true);

    auto matmul_value = op::MatMulV2("matmul_value");
    matmul_value.set_input_x1(value_x_data);
    matmul_value.set_input_x2(kernel_value_data);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(true);

    auto addn = op::AddN("addn");
    addn.create_dynamic_input_x(4);
    addn.set_dynamic_input_x(0, ln_bp, "pd_x");
    addn.set_dynamic_input_x(1, matmul_query);
    addn.set_dynamic_input_x(2, matmul_key);
    addn.set_dynamic_input_x(3, matmul_value);
    addn.set_attr_N(4);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    addn.update_output_desc_y(output_desc_y);

    std::vector<Operator> inputs{query_x_data};
    std::vector<Operator> outputs{addn};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 768, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradX") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}

TEST_F(attention_qkv_gradx_fusion_test, attention_qkv_gradx_fusion_test_8) {
    // success testcase
    ge::Graph graph("attention_qkv_gradx_fusion_test8");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // ln_bp
    auto ln_dy_input_data = op::Data("ln_dy_input_data");
    TensorDesc ln_dy_input_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    ln_dy_input_data.update_input_desc_x(ln_dy_input_desc);
    ln_dy_input_data.update_output_desc_y(ln_dy_input_desc);

    auto ln_x_input_data = op::Data("ln_x_input_data");
    TensorDesc ln_x_input_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    ln_x_input_data.update_input_desc_x(ln_x_input_desc);
    ln_x_input_data.update_output_desc_y(ln_x_input_desc);

    auto ln_mean_input_data = op::Data("ln_mean_input_data");
    TensorDesc ln_mean_input_desc(ge::Shape({12288}), FORMAT_NHWC, DT_FLOAT16);
    ln_mean_input_data.update_input_desc_x(ln_mean_input_desc);
    ln_mean_input_data.update_output_desc_y(ln_mean_input_desc);

    auto ln_variance_data = op::Data("ln_variance_data");
    TensorDesc ln_variance_input_desc(ge::Shape({12288}), FORMAT_NHWC, DT_FLOAT16);
    ln_variance_data.update_input_desc_x(ln_variance_input_desc);
    ln_variance_data.update_output_desc_y(ln_variance_input_desc);

    auto ln_gamma_data = op::Data("ln_gamma_data");
    TensorDesc ln_gamma_input_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    ln_gamma_data.update_input_desc_x(ln_gamma_input_desc);
    ln_gamma_data.update_output_desc_y(ln_gamma_input_desc);

    auto ln_bp = op::LayerNormXBackpropV2("ln_bp");
    ln_bp.set_input_dy(ln_dy_input_data);
    ln_bp.set_input_x(ln_x_input_data);
    ln_bp.set_input_mean(ln_mean_input_data);
    ln_bp.set_input_variance(ln_variance_data);
    ln_bp.set_input_gamma(ln_gamma_data);

    // matmul_dx_qkv
    auto query_x_data = op::Data("query_x_data");
    TensorDesc query_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    query_x_data.update_input_desc_x(query_x_desc);
    query_x_data.update_output_desc_y(query_x_desc);

    auto key_x_data = op::Data("key_x_data");
    TensorDesc key_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    key_x_data.update_input_desc_x(key_x_desc);
    key_x_data.update_output_desc_y(key_x_desc);

    auto value_x_data = op::Data("value_x_data");
    TensorDesc value_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    value_x_data.update_input_desc_x(value_x_desc);
    value_x_data.update_output_desc_y(value_x_desc);

    auto kernel_query_data = op::Data("kernel_query_data");
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_query_data.update_input_desc_x(kernel_query_desc);
    kernel_query_data.update_output_desc_y(kernel_query_desc);

    auto kernel_key_data = op::Data("kernel_key_data");
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_key_data.update_input_desc_x(kernel_key_desc);
    kernel_key_data.update_output_desc_y(kernel_key_desc);

    auto kernel_value_data = op::Data("kernel_value_data");
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_value_data.update_input_desc_x(kernel_value_desc);
    kernel_value_data.update_output_desc_y(kernel_value_desc);

    auto matmul_query = op::MatMulV2("matmul_query");
    matmul_query.set_input_x1(query_x_data);
    matmul_query.set_input_x2(kernel_query_data);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(true);

    auto matmul_key = op::MatMulV2("matmul_key");
    matmul_key.set_input_x1(key_x_data);
    matmul_key.set_input_x2(kernel_key_data);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(true);

    auto matmul_value = op::MatMulV2("matmul_value");
    matmul_value.set_input_x1(value_x_data);
    matmul_value.set_input_x2(kernel_value_data);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(true);

    auto addn = op::AddN("addn");
    addn.create_dynamic_input_x(4);
    addn.set_dynamic_input_x(0, ln_bp, "pd_x");
    addn.set_dynamic_input_x(1, matmul_query);
    addn.set_dynamic_input_x(2, matmul_key);
    addn.set_dynamic_input_x(3, matmul_value);
    addn.set_attr_N(4);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_NHWC, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    addn.update_output_desc_y(output_desc_y);

    std::vector<Operator> inputs{query_x_data};
    std::vector<Operator> outputs{addn};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 768, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradX") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}