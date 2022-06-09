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
#include "reduce_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/operator_reg.h"
#include "fc_transdata_merge_fusion_pass_test.h"
#define private public
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;

class attention_qkv_gradw_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "attention_qkv_gradw_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "attention_qkv_gradw_fusion_test TearDown" << std::endl;
  }
};

TEST_F(attention_qkv_gradw_fusion_test, attention_qkv_gradw_fusion_test_1) {
    // success testcase
    ge::Graph graph("attention_qkv_gradw_fusion_test1");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // Batchmatmul_dw
    auto bmm_x_data = op::Data("bmm_x_data");
    TensorDesc bmm_x_desc(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    bmm_x_data.update_input_desc_x(bmm_x_desc);
    bmm_x_data.update_output_desc_y(bmm_x_desc);

    auto kernel_bmm_data = op::Data("kernel_bmm_data");
    TensorDesc kernel_bmm_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_bmm_data.update_input_desc_x(kernel_bmm_desc);
    kernel_bmm_data.update_output_desc_y(kernel_bmm_desc);

    auto bmm = op::BatchMatMulV2("bmm");
    bmm.set_input_x1(bmm_x_data);
    bmm.set_input_x2(kernel_bmm_data);
    bmm.set_attr_adj_x1(false);
    bmm.set_attr_adj_x2(false);
    bmm.set_attr_offset_x(0);

    // conf_transpose
    auto conf_trans_dw_query = op::ConfusionTransposeD("conf_trans_dw_query");
    conf_trans_dw_query.set_input_x(bmm);
    conf_trans_dw_query.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_query.set_attr_shape({12288, 1024});
    conf_trans_dw_query.set_attr_transpose_first(1);
    TensorDesc output_desc_query(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_query.SetOriginShape(ge::Shape({12288, 1024}));
    conf_trans_dw_query.update_output_desc_y(output_desc_query);

    auto conf_trans_dw_key = op::ConfusionTransposeD("conf_trans_dw_key");
    conf_trans_dw_key.set_input_x(bmm);
    conf_trans_dw_key.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_key.set_attr_shape({12288, 1024});
    conf_trans_dw_key.set_attr_transpose_first(1);

    auto conf_trans_dw_value = op::ConfusionTransposeD("conf_trans_dw_value");
    conf_trans_dw_value.set_input_x(bmm);
    conf_trans_dw_value.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_value.set_attr_shape({12288, 1024});
    conf_trans_dw_value.set_attr_transpose_first(1);

    // layer_norm
    auto input_x_data = op::Data("input_x_data");
    TensorDesc input_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_x_data.update_input_desc_x(input_x_desc);
    input_x_data.update_output_desc_y(input_x_desc);

    TensorDesc gamma_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor gamma_tensor(gamma_desc);
    auto gamma = op::Const("gamma").set_attr_value(gamma_tensor);

    TensorDesc beta_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor beta_tensor(beta_desc);
    auto beta = op::Const("beta").set_attr_value(beta_tensor);

    auto layer_norm = op::LayerNorm("layer_norm_0");
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    // matmul_dx_qkv
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    auto matmul_dx_query = op::MatMulV2("matmul_dx_query");
    matmul_dx_query.set_input_x1(conf_trans_dw_query);
    matmul_dx_query.set_input_x2(kernel_query);

    auto matmul_dx_key = op::MatMulV2("matmul_dx_key");
    matmul_dx_key.set_input_x1(conf_trans_dw_key);
    matmul_dx_key.set_input_x2(kernel_key);

    auto matmul_dx_value = op::MatMulV2("matmul_dx_value");
    matmul_dx_value.set_input_x1(conf_trans_dw_value);
    matmul_dx_value.set_input_x2(kernel_value);

    // matmul_dw_qkv
    auto matmul_dw_query = op::MatMulV2("matmul_dw_query");
    matmul_dw_query.set_input_x1(layer_norm, "y");
    matmul_dw_query.set_input_x2(conf_trans_dw_query);
    matmul_dw_query.set_attr_transpose_x1(true);
    matmul_dw_query.set_attr_transpose_x2(false);
    TensorDesc output_desc_dw_query(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_dw_query.update_output_desc_y(output_desc_dw_query);

    auto matmul_dw_key = op::MatMulV2("matmul_dw_key");
    matmul_dw_key.set_input_x1(layer_norm, "y");
    matmul_dw_key.set_input_x2(conf_trans_dw_key);
    matmul_dw_key.set_attr_transpose_x1(true);
    matmul_dw_key.set_attr_transpose_x2(false);
    TensorDesc output_desc_dw_key(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_dw_key.update_output_desc_y(output_desc_dw_key);

    auto matmul_dw_value = op::MatMulV2("matmul_dw_value");
    matmul_dw_value.set_input_x1(layer_norm, "y");
    matmul_dw_value.set_input_x2(conf_trans_dw_value);
    matmul_dw_value.set_attr_transpose_x1(true);
    matmul_dw_value.set_attr_transpose_x2(false);
    TensorDesc output_desc_dw_value(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_dw_value.update_output_desc_y(output_desc_dw_value);

    // reduce_sum_qkv
    std::vector<int64_t> axes_value{1, 2};
    auto reduce_sum_query = op::ReduceSumD("reduce_sum_query");
    reduce_sum_query.set_input_x(conf_trans_dw_query);
    reduce_sum_query.set_attr_axes(axes_value);
    reduce_sum_query.set_attr_keep_dims(false);

    auto reduce_sum_key = op::ReduceSumD("reduce_sum_key");
    reduce_sum_key.set_input_x(conf_trans_dw_key);
    reduce_sum_key.set_attr_axes(axes_value);
    reduce_sum_key.set_attr_keep_dims(false);

    auto reduce_sum_value = op::ReduceSumD("reduce_sum_value");
    reduce_sum_value.set_input_x(conf_trans_dw_value);
    reduce_sum_value.set_attr_axes(axes_value);
    reduce_sum_value.set_attr_keep_dims(false);

    std::vector<Operator> inputs{bmm_x_data, kernel_bmm_data};
    std::vector<Operator> outputs{matmul_dw_query, matmul_dw_key, matmul_dw_value, reduce_sum_query, reduce_sum_key, reduce_sum_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradWFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 64, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradW") {
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

TEST_F(attention_qkv_gradw_fusion_test, attention_qkv_gradw_fusion_test_2) {
    // num of conf_transpose_node outputs less than 3
    ge::Graph graph("attention_qkv_gradw_fusion_test2");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // Batchmatmul_dw
    auto bmm_x_data = op::Data("bmm_x_data");
    TensorDesc bmm_x_desc(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    bmm_x_data.update_input_desc_x(bmm_x_desc);
    bmm_x_data.update_output_desc_y(bmm_x_desc);

    auto kernel_bmm_data = op::Data("kernel_bmm_data");
    TensorDesc kernel_bmm_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_bmm_data.update_input_desc_x(kernel_bmm_desc);
    kernel_bmm_data.update_output_desc_y(kernel_bmm_desc);

    auto bmm = op::BatchMatMulV2("bmm");
    bmm.set_input_x1(bmm_x_data);
    bmm.set_input_x2(kernel_bmm_data);
    bmm.set_attr_adj_x1(false);
    bmm.set_attr_adj_x2(false);
    bmm.set_attr_offset_x(0);

    // conf_transpose
    auto conf_trans_dw_query = op::ConfusionTransposeD("conf_trans_dw_query");
    conf_trans_dw_query.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_query.set_attr_shape({12288, 1024});
    conf_trans_dw_query.set_attr_transpose_first(1);
    conf_trans_dw_query.set_input_x(bmm);
    TensorDesc output_desc_query(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_query.SetOriginShape(ge::Shape({12288, 1024}));
    conf_trans_dw_query.update_output_desc_y(output_desc_query);

    auto conf_trans_dw_key = op::ConfusionTransposeD("conf_trans_dw_key");
    conf_trans_dw_key.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_key.set_attr_shape({12288, 1024});
    conf_trans_dw_key.set_attr_transpose_first(1);
    conf_trans_dw_key.set_input_x(bmm);

    auto conf_trans_dw_value = op::ConfusionTransposeD("conf_trans_dw_value");
    conf_trans_dw_value.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_value.set_attr_shape({12288, 1024});
    conf_trans_dw_value.set_attr_transpose_first(1);
    conf_trans_dw_value.set_input_x(bmm);

    // layer_norm
    auto input_x_data = op::Data("input_x_data");
    TensorDesc input_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_x_data.update_input_desc_x(input_x_desc);
    input_x_data.update_output_desc_y(input_x_desc);

    TensorDesc gamma_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor gamma_tensor(gamma_desc);
    auto gamma = op::Const("gamma").set_attr_value(gamma_tensor);

    TensorDesc beta_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor beta_tensor(beta_desc);
    auto beta = op::Const("beta").set_attr_value(beta_tensor);

    auto layer_norm = op::LayerNorm("layer_norm_0");
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    // matmul_dw_qkv
    auto matmul_dw_query = op::MatMulV2("matmul_dw_query");
    matmul_dw_query.set_input_x1(layer_norm, "y");
    matmul_dw_query.set_input_x2(conf_trans_dw_query);
    matmul_dw_query.set_attr_transpose_x1(true);
    matmul_dw_query.set_attr_transpose_x2(false);

    auto matmul_dw_key = op::MatMulV2("matmul_dw_key");
    matmul_dw_key.set_input_x1(layer_norm, "y");
    matmul_dw_key.set_input_x2(conf_trans_dw_key);
    matmul_dw_key.set_attr_transpose_x1(true);
    matmul_dw_key.set_attr_transpose_x2(false);

    auto matmul_dw_value = op::MatMulV2("matmul_dw_value");
    matmul_dw_value.set_input_x1(layer_norm, "y");
    matmul_dw_value.set_input_x2(conf_trans_dw_value);
    matmul_dw_value.set_attr_transpose_x1(true);
    matmul_dw_value.set_attr_transpose_x2(false);

    // reduce_sum_qkv
    std::vector<int64_t> axes_value = {0};
    auto reduce_sum_query = op::ReduceSumD("reduce_sum_query");
    reduce_sum_query.set_input_x(conf_trans_dw_query);
    reduce_sum_query.set_attr_axes(axes_value);
    reduce_sum_query.set_attr_keep_dims(false);

    auto reduce_sum_key = op::ReduceSumD("reduce_sum_key");
    reduce_sum_key.set_input_x(conf_trans_dw_key);
    reduce_sum_key.set_attr_axes(axes_value);
    reduce_sum_key.set_attr_keep_dims(false);

    auto reduce_sum_value = op::ReduceSumD("reduce_sum_value");
    reduce_sum_value.set_input_x(conf_trans_dw_value);
    reduce_sum_value.set_attr_axes(axes_value);
    reduce_sum_value.set_attr_keep_dims(false);

    std::vector<Operator> inputs{bmm_x_data};
    std::vector<Operator> outputs{matmul_dw_query, matmul_dw_key, matmul_dw_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradWFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 64, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradW") {
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

TEST_F(attention_qkv_gradw_fusion_test, attention_qkv_gradw_fusion_test_3) {
    // num of matmul_dw less than 3
    ge::Graph graph("attention_qkv_gradw_fusion_test3");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // Batchmatmul_dw
    auto bmm_x_data = op::Data("bmm_x_data");
    TensorDesc bmm_x_desc(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    bmm_x_data.update_input_desc_x(bmm_x_desc);
    bmm_x_data.update_output_desc_y(bmm_x_desc);

    auto kernel_bmm_data = op::Data("kernel_bmm_data");
    TensorDesc kernel_bmm_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_bmm_data.update_input_desc_x(kernel_bmm_desc);
    kernel_bmm_data.update_output_desc_y(kernel_bmm_desc);

    auto bmm = op::BatchMatMulV2("bmm");
    bmm.set_input_x1(bmm_x_data);
    bmm.set_input_x2(kernel_bmm_data);
    bmm.set_attr_adj_x1(false);
    bmm.set_attr_adj_x2(false);
    bmm.set_attr_offset_x(0);

    // conf_transpose
    auto conf_trans_dw_query = op::ConfusionTransposeD("conf_trans_dw_query");
    conf_trans_dw_query.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_query.set_attr_shape({12288, 1024});
    conf_trans_dw_query.set_attr_transpose_first(1);
    conf_trans_dw_query.set_input_x(bmm);
    TensorDesc output_desc_query(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_query.SetOriginShape(ge::Shape({12288, 1024}));
    conf_trans_dw_query.update_output_desc_y(output_desc_query);

    auto conf_trans_dw_key = op::ConfusionTransposeD("conf_trans_dw_key");
    conf_trans_dw_key.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_key.set_attr_shape({12288, 1024});
    conf_trans_dw_key.set_attr_transpose_first(1);
    conf_trans_dw_key.set_input_x(bmm);

    auto conf_trans_dw_value = op::ConfusionTransposeD("conf_trans_dw_value");
    conf_trans_dw_value.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_value.set_attr_shape({12288, 1024});
    conf_trans_dw_value.set_attr_transpose_first(1);
    conf_trans_dw_value.set_input_x(bmm);

    // layer_norm
    auto input_x_data = op::Data("input_x_data");
    TensorDesc input_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_x_data.update_input_desc_x(input_x_desc);
    input_x_data.update_output_desc_y(input_x_desc);

    TensorDesc gamma_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor gamma_tensor(gamma_desc);
    auto gamma = op::Const("gamma").set_attr_value(gamma_tensor);

    TensorDesc beta_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor beta_tensor(beta_desc);
    auto beta = op::Const("beta").set_attr_value(beta_tensor);

    auto layer_norm = op::LayerNorm("layer_norm_0");
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    // matmul_dx_qkv
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    auto matmul_dx_query = op::MatMulV2("matmul_dx_query");
    matmul_dx_query.set_input_x1(conf_trans_dw_query);
    matmul_dx_query.set_input_x2(kernel_query);

    auto matmul_dx_key = op::MatMulV2("matmul_dx_key");
    matmul_dx_key.set_input_x1(conf_trans_dw_key);
    matmul_dx_key.set_input_x2(kernel_key);

    auto matmul_dx_value = op::MatMulV2("matmul_dx_value");
    matmul_dx_value.set_input_x1(conf_trans_dw_value);
    matmul_dx_value.set_input_x2(kernel_value);

    // matmul_dw_qkv
    auto matmul_dw_query = op::MatMulV2("matmul_dw_query");
    matmul_dw_query.set_input_x1(layer_norm, "y");
    matmul_dw_query.set_input_x2(conf_trans_dw_query);
    matmul_dw_query.set_attr_transpose_x1(true);
    matmul_dw_query.set_attr_transpose_x2(false);

    auto matmul_dw_key = op::MatMulV2("matmul_dw_key");
    matmul_dw_key.set_input_x1(layer_norm, "y");
    matmul_dw_key.set_input_x2(conf_trans_dw_key);
    matmul_dw_key.set_attr_transpose_x1(true);
    matmul_dw_key.set_attr_transpose_x2(false);

    // reduce_sum_qkv
    std::vector<int64_t> axes_value = {0};
    auto reduce_sum_query = op::ReduceSumD("reduce_sum_query");
    reduce_sum_query.set_input_x(conf_trans_dw_query);
    reduce_sum_query.set_attr_axes(axes_value);
    reduce_sum_query.set_attr_keep_dims(false);

    auto reduce_sum_key = op::ReduceSumD("reduce_sum_key");
    reduce_sum_key.set_input_x(conf_trans_dw_key);
    reduce_sum_key.set_attr_axes(axes_value);
    reduce_sum_key.set_attr_keep_dims(false);

    auto reduce_sum_value = op::ReduceSumD("reduce_sum_value");
    reduce_sum_value.set_input_x(conf_trans_dw_value);
    reduce_sum_value.set_attr_axes(axes_value);
    reduce_sum_value.set_attr_keep_dims(false);

    std::vector<Operator> inputs{bmm_x_data};
    std::vector<Operator> outputs{matmul_dw_query, matmul_dw_key};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradWFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 64, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradW") {
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

TEST_F(attention_qkv_gradw_fusion_test, attention_qkv_gradw_fusion_test_4) {
    // conf_transpose_node output is not MatMulV2
    ge::Graph graph("attention_qkv_gradw_fusion_test4");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // Batchmatmul_dw
    auto bmm_x_data = op::Data("bmm_x_data");
    TensorDesc bmm_x_desc(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    bmm_x_data.update_input_desc_x(bmm_x_desc);
    bmm_x_data.update_output_desc_y(bmm_x_desc);

    auto kernel_bmm_data = op::Data("kernel_bmm_data");
    TensorDesc kernel_bmm_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_bmm_data.update_input_desc_x(kernel_bmm_desc);
    kernel_bmm_data.update_output_desc_y(kernel_bmm_desc);

    auto bmm = op::BatchMatMulV2("bmm");
    bmm.set_input_x1(bmm_x_data);
    bmm.set_input_x2(kernel_bmm_data);
    bmm.set_attr_adj_x1(false);
    bmm.set_attr_adj_x2(false);
    bmm.set_attr_offset_x(0);

    // conf_transpose
    auto conf_trans_dw_query = op::ConfusionTransposeD("conf_trans_dw_query");
    conf_trans_dw_query.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_query.set_attr_shape({12288, 1024});
    conf_trans_dw_query.set_attr_transpose_first(1);
    conf_trans_dw_query.set_input_x(bmm);
    TensorDesc output_desc_query(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_query.SetOriginShape(ge::Shape({12288, 1024}));
    conf_trans_dw_query.update_output_desc_y(output_desc_query);

    auto conf_trans_dw_key = op::ConfusionTransposeD("conf_trans_dw_key");
    conf_trans_dw_key.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_key.set_attr_shape({12288, 1024});
    conf_trans_dw_key.set_attr_transpose_first(1);
    conf_trans_dw_key.set_input_x(bmm);

    auto conf_trans_dw_value = op::ConfusionTransposeD("conf_trans_dw_value");
    conf_trans_dw_value.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_value.set_attr_shape({12288, 1024});
    conf_trans_dw_value.set_attr_transpose_first(1);
    conf_trans_dw_value.set_input_x(bmm);

    // layer_norm
    auto input_x_data = op::Data("input_x_data");
    TensorDesc input_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_x_data.update_input_desc_x(input_x_desc);
    input_x_data.update_output_desc_y(input_x_desc);

    TensorDesc gamma_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor gamma_tensor(gamma_desc);
    auto gamma = op::Const("gamma").set_attr_value(gamma_tensor);

    TensorDesc beta_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor beta_tensor(beta_desc);
    auto beta = op::Const("beta").set_attr_value(beta_tensor);

    auto layer_norm = op::LayerNorm("layer_norm_0");
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    // matmul_dx_qkv
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    auto matmul_dx_query = op::MatMulV2("matmul_dx_query");
    matmul_dx_query.set_input_x1(conf_trans_dw_query);
    matmul_dx_query.set_input_x2(kernel_query);

    auto matmul_dx_key = op::MatMulV2("matmul_dx_key");
    matmul_dx_key.set_input_x1(conf_trans_dw_key);
    matmul_dx_key.set_input_x2(kernel_key);

    auto matmul_dx_value = op::MatMulV2("matmul_dx_value");
    matmul_dx_value.set_input_x1(conf_trans_dw_value);
    matmul_dx_value.set_input_x2(kernel_value);

    // matmul_dw_qkv
    auto matmul_dw_query = op::MatMul("matmul_dw_query");
    matmul_dw_query.set_input_x1(layer_norm, "y");
    matmul_dw_query.set_input_x2(conf_trans_dw_query);
    matmul_dw_query.set_attr_transpose_x1(true);
    matmul_dw_query.set_attr_transpose_x2(false);

    auto matmul_dw_key = op::MatMulV2("matmul_dw_key");
    matmul_dw_key.set_input_x1(layer_norm, "y");
    matmul_dw_key.set_input_x2(conf_trans_dw_key);
    matmul_dw_key.set_attr_transpose_x1(true);
    matmul_dw_key.set_attr_transpose_x2(false);

    auto matmul_dw_value = op::MatMulV2("matmul_dw_value");
    matmul_dw_value.set_input_x1(layer_norm, "y");
    matmul_dw_value.set_input_x2(conf_trans_dw_value);
    matmul_dw_value.set_attr_transpose_x1(true);
    matmul_dw_value.set_attr_transpose_x2(false);

    // reduce_sum_qkv
    std::vector<int64_t> axes_value = {0};
    auto reduce_sum_query = op::ReduceSumD("reduce_sum_query");
    reduce_sum_query.set_input_x(conf_trans_dw_query);
    reduce_sum_query.set_attr_axes(axes_value);
    reduce_sum_query.set_attr_keep_dims(false);

    auto reduce_sum_key = op::ReduceSumD("reduce_sum_key");
    reduce_sum_key.set_input_x(conf_trans_dw_key);
    reduce_sum_key.set_attr_axes(axes_value);
    reduce_sum_key.set_attr_keep_dims(false);

    auto reduce_sum_value = op::ReduceSumD("reduce_sum_value");
    reduce_sum_value.set_input_x(conf_trans_dw_value);
    reduce_sum_value.set_attr_axes(axes_value);
    reduce_sum_value.set_attr_keep_dims(false);

    std::vector<Operator> inputs{bmm_x_data};
    std::vector<Operator> outputs{matmul_dw_query, matmul_dw_key, matmul_dw_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradWFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 64, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradW") {
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

TEST_F(attention_qkv_gradw_fusion_test, attention_qkv_gradw_fusion_test_5) {
    // matmul_dw input is not conf_transpose_node
    ge::Graph graph("attention_qkv_gradw_fusion_test5");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // Batchmatmul_dw
    auto bmm_x_data = op::Data("bmm_x_data");
    TensorDesc bmm_x_desc(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    bmm_x_data.update_input_desc_x(bmm_x_desc);
    bmm_x_data.update_output_desc_y(bmm_x_desc);

    auto kernel_bmm_data = op::Data("kernel_bmm_data");
    TensorDesc kernel_bmm_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_bmm_data.update_input_desc_x(kernel_bmm_desc);
    kernel_bmm_data.update_output_desc_y(kernel_bmm_desc);

    auto bmm = op::BatchMatMulV2("bmm");
    bmm.set_input_x1(bmm_x_data);
    bmm.set_input_x2(kernel_bmm_data);
    bmm.set_attr_adj_x1(false);
    bmm.set_attr_adj_x2(false);
    bmm.set_attr_offset_x(0);

    // conf_transpose
    auto conf_trans_dw_query = op::ConfusionTransposeD("conf_trans_dw_query");
    conf_trans_dw_query.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_query.set_attr_shape({12288, 1024});
    conf_trans_dw_query.set_attr_transpose_first(1);
    conf_trans_dw_query.set_input_x(bmm);
    TensorDesc output_desc_query(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_query.SetOriginShape(ge::Shape({12288, 1024}));
    conf_trans_dw_query.update_output_desc_y(output_desc_query);

    auto conf_trans_dw_key = op::ConfusionTransposeD("conf_trans_dw_key");
    conf_trans_dw_key.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_key.set_attr_shape({12288, 1024});
    conf_trans_dw_key.set_attr_transpose_first(1);
    conf_trans_dw_key.set_input_x(bmm);

    auto conf_trans_dw_value = op::ReFormat("conf_trans_dw_value");
    TensorDesc reformat_in_desc(ge::Shape({12288, 1024}), FORMAT_ND, DT_FLOAT16);
    TensorDesc reformat_out_desc(ge::Shape({12288, 1024}), FORMAT_NHWC, DT_FLOAT16);
    conf_trans_dw_value.update_input_desc_x(reformat_in_desc);
    conf_trans_dw_value.update_output_desc_y(reformat_out_desc);
    conf_trans_dw_value.set_input_x(bmm);

    // layer_norm
    auto input_x_data = op::Data("input_x_data");
    TensorDesc input_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_x_data.update_input_desc_x(input_x_desc);
    input_x_data.update_output_desc_y(input_x_desc);

    TensorDesc gamma_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor gamma_tensor(gamma_desc);
    auto gamma = op::Const("gamma").set_attr_value(gamma_tensor);

    TensorDesc beta_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor beta_tensor(beta_desc);
    auto beta = op::Const("beta").set_attr_value(beta_tensor);

    auto layer_norm = op::LayerNorm("layer_norm_0");
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    // matmul_dx_qkv
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    auto matmul_dx_query = op::MatMulV2("matmul_dx_query");
    matmul_dx_query.set_input_x1(conf_trans_dw_query);
    matmul_dx_query.set_input_x2(kernel_query);

    auto matmul_dx_key = op::MatMulV2("matmul_dx_key");
    matmul_dx_key.set_input_x1(conf_trans_dw_key);
    matmul_dx_key.set_input_x2(kernel_key);

    auto matmul_dx_value = op::MatMulV2("matmul_dx_value");
    matmul_dx_value.set_input_x1(conf_trans_dw_value);
    matmul_dx_value.set_input_x2(kernel_value);

    // matmul_dw_qkv
    auto matmul_dw_query = op::MatMulV2("matmul_dw_query");
    matmul_dw_query.set_input_x1(layer_norm, "y");
    matmul_dw_query.set_input_x2(conf_trans_dw_query);
    matmul_dw_query.set_attr_transpose_x1(true);
    matmul_dw_query.set_attr_transpose_x2(false);

    auto matmul_dw_key = op::MatMulV2("matmul_dw_key");
    matmul_dw_key.set_input_x1(layer_norm, "y");
    matmul_dw_key.set_input_x2(conf_trans_dw_key);
    matmul_dw_key.set_attr_transpose_x1(true);
    matmul_dw_key.set_attr_transpose_x2(false);

    auto matmul_dw_value = op::MatMulV2("matmul_dw_value");
    matmul_dw_value.set_input_x1(layer_norm, "y");
    matmul_dw_value.set_input_x2(conf_trans_dw_value);
    matmul_dw_value.set_attr_transpose_x1(true);
    matmul_dw_value.set_attr_transpose_x2(false);

    // reduce_sum_qkv
    std::vector<int64_t> axes_value = {0};
    auto reduce_sum_query = op::ReduceSumD("reduce_sum_query");
    reduce_sum_query.set_input_x(conf_trans_dw_query);
    reduce_sum_query.set_attr_axes(axes_value);
    reduce_sum_query.set_attr_keep_dims(false);

    auto reduce_sum_key = op::ReduceSumD("reduce_sum_key");
    reduce_sum_key.set_input_x(conf_trans_dw_key);
    reduce_sum_key.set_attr_axes(axes_value);
    reduce_sum_key.set_attr_keep_dims(false);

    auto reduce_sum_value = op::ReduceSumD("reduce_sum_value");
    reduce_sum_value.set_input_x(conf_trans_dw_value);
    reduce_sum_value.set_attr_axes(axes_value);
    reduce_sum_value.set_attr_keep_dims(false);

    std::vector<Operator> inputs{bmm_x_data};
    std::vector<Operator> outputs{matmul_dw_query, matmul_dw_key, matmul_dw_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradWFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 64, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradW") {
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

TEST_F(attention_qkv_gradw_fusion_test, attention_qkv_gradw_fusion_test_6) {
    // trans_flag match failed
    ge::Graph graph("attention_qkv_gradw_fusion_test6");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // Batchmatmul_dw
    auto bmm_x_data = op::Data("bmm_x_data");
    TensorDesc bmm_x_desc(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    bmm_x_data.update_input_desc_x(bmm_x_desc);
    bmm_x_data.update_output_desc_y(bmm_x_desc);

    auto kernel_bmm_data = op::Data("kernel_bmm_data");
    TensorDesc kernel_bmm_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_bmm_data.update_input_desc_x(kernel_bmm_desc);
    kernel_bmm_data.update_output_desc_y(kernel_bmm_desc);

    auto bmm = op::BatchMatMulV2("bmm");
    bmm.set_input_x1(bmm_x_data);
    bmm.set_input_x2(kernel_bmm_data);
    bmm.set_attr_adj_x1(false);
    bmm.set_attr_adj_x2(false);
    bmm.set_attr_offset_x(0);

    // conf_transpose
    auto conf_trans_dw_query = op::ConfusionTransposeD("conf_trans_dw_query");
    conf_trans_dw_query.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_query.set_attr_shape({12288, 1024});
    conf_trans_dw_query.set_attr_transpose_first(1);
    conf_trans_dw_query.set_input_x(bmm);
    TensorDesc output_desc_query(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_query.SetOriginShape(ge::Shape({12288, 1024}));
    conf_trans_dw_query.update_output_desc_y(output_desc_query);

    auto conf_trans_dw_key = op::ConfusionTransposeD("conf_trans_dw_key");
    conf_trans_dw_key.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_key.set_attr_shape({12288, 1024});
    conf_trans_dw_key.set_attr_transpose_first(1);
    conf_trans_dw_key.set_input_x(bmm);

    auto conf_trans_dw_value = op::ConfusionTransposeD("conf_trans_dw_value");
    conf_trans_dw_value.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_value.set_attr_shape({12288, 1024});
    conf_trans_dw_value.set_attr_transpose_first(1);
    conf_trans_dw_value.set_input_x(bmm);

    // layer_norm
    auto input_x_data = op::Data("input_x_data");
    TensorDesc input_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_x_data.update_input_desc_x(input_x_desc);
    input_x_data.update_output_desc_y(input_x_desc);

    TensorDesc gamma_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor gamma_tensor(gamma_desc);
    auto gamma = op::Const("gamma").set_attr_value(gamma_tensor);

    TensorDesc beta_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor beta_tensor(beta_desc);
    auto beta = op::Const("beta").set_attr_value(beta_tensor);

    auto layer_norm = op::LayerNorm("layer_norm_0");
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    // matmul_dx_qkv
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    auto matmul_dx_query = op::MatMulV2("matmul_dx_query");
    matmul_dx_query.set_input_x1(conf_trans_dw_query);
    matmul_dx_query.set_input_x2(kernel_query);

    auto matmul_dx_key = op::MatMulV2("matmul_dx_key");
    matmul_dx_key.set_input_x1(conf_trans_dw_key);
    matmul_dx_key.set_input_x2(kernel_key);

    auto matmul_dx_value = op::MatMulV2("matmul_dx_value");
    matmul_dx_value.set_input_x1(conf_trans_dw_value);
    matmul_dx_value.set_input_x2(kernel_value);

    // matmul_dw_qkv
    auto matmul_dw_query = op::MatMulV2("matmul_dw_query");
    matmul_dw_query.set_input_x1(layer_norm, "y");
    matmul_dw_query.set_input_x2(conf_trans_dw_query);
    matmul_dw_query.set_attr_transpose_x1(false);
    matmul_dw_query.set_attr_transpose_x2(false);

    auto matmul_dw_key = op::MatMulV2("matmul_dw_key");
    matmul_dw_key.set_input_x1(layer_norm, "y");
    matmul_dw_key.set_input_x2(conf_trans_dw_key);
    matmul_dw_key.set_attr_transpose_x1(true);
    matmul_dw_key.set_attr_transpose_x2(false);

    auto matmul_dw_value = op::MatMulV2("matmul_dw_value");
    matmul_dw_value.set_input_x1(layer_norm, "y");
    matmul_dw_value.set_input_x2(conf_trans_dw_value);
    matmul_dw_value.set_attr_transpose_x1(true);
    matmul_dw_value.set_attr_transpose_x2(false);

    // reduce_sum_qkv
    std::vector<int64_t> axes_value = {0};
    auto reduce_sum_query = op::ReduceSumD("reduce_sum_query");
    reduce_sum_query.set_input_x(conf_trans_dw_query);
    reduce_sum_query.set_attr_axes(axes_value);
    reduce_sum_query.set_attr_keep_dims(false);

    auto reduce_sum_key = op::ReduceSumD("reduce_sum_key");
    reduce_sum_key.set_input_x(conf_trans_dw_key);
    reduce_sum_key.set_attr_axes(axes_value);
    reduce_sum_key.set_attr_keep_dims(false);

    auto reduce_sum_value = op::ReduceSumD("reduce_sum_value");
    reduce_sum_value.set_input_x(conf_trans_dw_value);
    reduce_sum_value.set_attr_axes(axes_value);
    reduce_sum_value.set_attr_keep_dims(false);

    std::vector<Operator> inputs{bmm_x_data};
    std::vector<Operator> outputs{matmul_dw_query, matmul_dw_key, matmul_dw_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradWFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 64, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradW") {
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

TEST_F(attention_qkv_gradw_fusion_test, attention_qkv_gradw_fusion_test_7) {
    // shape match failed
    ge::Graph graph("attention_qkv_gradw_fusion_test7");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // Batchmatmul_dw
    auto bmm_x_data = op::Data("bmm_x_data");
    TensorDesc bmm_x_desc(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    bmm_x_data.update_input_desc_x(bmm_x_desc);
    bmm_x_data.update_output_desc_y(bmm_x_desc);

    auto kernel_bmm_data = op::Data("kernel_bmm_data");
    TensorDesc kernel_bmm_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_bmm_data.update_input_desc_x(kernel_bmm_desc);
    kernel_bmm_data.update_output_desc_y(kernel_bmm_desc);

    auto bmm = op::BatchMatMulV2("bmm");
    bmm.set_input_x1(bmm_x_data);
    bmm.set_input_x2(kernel_bmm_data);
    bmm.set_attr_adj_x1(false);
    bmm.set_attr_adj_x2(false);
    bmm.set_attr_offset_x(0);

    // conf_transpose
    auto conf_trans_dw_query = op::ConfusionTransposeD("conf_trans_dw_query");
    conf_trans_dw_query.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_query.set_attr_shape({12288, 768});
    conf_trans_dw_query.set_attr_transpose_first(1);
    conf_trans_dw_query.set_input_x(bmm);
    TensorDesc output_desc_query(ge::Shape({48, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_query.SetOriginShape(ge::Shape({12288, 768}));
    conf_trans_dw_query.update_output_desc_y(output_desc_query);

    auto conf_trans_dw_key = op::ConfusionTransposeD("conf_trans_dw_key");
    conf_trans_dw_key.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_key.set_attr_shape({12288, 1024});
    conf_trans_dw_key.set_attr_transpose_first(1);
    conf_trans_dw_key.set_input_x(bmm);

    auto conf_trans_dw_value = op::ConfusionTransposeD("conf_trans_dw_value");
    conf_trans_dw_value.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_value.set_attr_shape({12288, 1024});
    conf_trans_dw_value.set_attr_transpose_first(1);
    conf_trans_dw_value.set_input_x(bmm);

    // layer_norm
    auto input_x_data = op::Data("input_x_data");
    TensorDesc input_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_x_data.update_input_desc_x(input_x_desc);
    input_x_data.update_output_desc_y(input_x_desc);

    TensorDesc gamma_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor gamma_tensor(gamma_desc);
    auto gamma = op::Const("gamma").set_attr_value(gamma_tensor);

    TensorDesc beta_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor beta_tensor(beta_desc);
    auto beta = op::Const("beta").set_attr_value(beta_tensor);

    auto layer_norm = op::LayerNorm("layer_norm_0");
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    TensorDesc output_desc_y(ge::Shape({48, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 768}));
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    // matmul_dx_qkv
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    auto matmul_dx_query = op::MatMulV2("matmul_dx_query");
    matmul_dx_query.set_input_x1(conf_trans_dw_query);
    matmul_dx_query.set_input_x2(kernel_query);

    auto matmul_dx_key = op::MatMulV2("matmul_dx_key");
    matmul_dx_key.set_input_x1(conf_trans_dw_key);
    matmul_dx_key.set_input_x2(kernel_key);

    auto matmul_dx_value = op::MatMulV2("matmul_dx_value");
    matmul_dx_value.set_input_x1(conf_trans_dw_value);
    matmul_dx_value.set_input_x2(kernel_value);

    // matmul_dw_qkv
    auto matmul_dw_query = op::MatMulV2("matmul_dw_query");
    matmul_dw_query.set_input_x1(layer_norm, "y");
    matmul_dw_query.set_input_x2(conf_trans_dw_query);
    matmul_dw_query.set_attr_transpose_x1(true);
    matmul_dw_query.set_attr_transpose_x2(false);

    auto matmul_dw_key = op::MatMulV2("matmul_dw_key");
    matmul_dw_key.set_input_x1(layer_norm, "y");
    matmul_dw_key.set_input_x2(conf_trans_dw_key);
    matmul_dw_key.set_attr_transpose_x1(true);
    matmul_dw_key.set_attr_transpose_x2(false);

    auto matmul_dw_value = op::MatMulV2("matmul_dw_value");
    matmul_dw_value.set_input_x1(layer_norm, "y");
    matmul_dw_value.set_input_x2(conf_trans_dw_value);
    matmul_dw_value.set_attr_transpose_x1(true);
    matmul_dw_value.set_attr_transpose_x2(false);

    // reduce_sum_qkv
    std::vector<int64_t> axes_value = {0};
    auto reduce_sum_query = op::ReduceSumD("reduce_sum_query");
    reduce_sum_query.set_input_x(conf_trans_dw_query);
    reduce_sum_query.set_attr_axes(axes_value);
    reduce_sum_query.set_attr_keep_dims(false);

    auto reduce_sum_key = op::ReduceSumD("reduce_sum_key");
    reduce_sum_key.set_input_x(conf_trans_dw_key);
    reduce_sum_key.set_attr_axes(axes_value);
    reduce_sum_key.set_attr_keep_dims(false);

    auto reduce_sum_value = op::ReduceSumD("reduce_sum_value");
    reduce_sum_value.set_input_x(conf_trans_dw_value);
    reduce_sum_value.set_attr_axes(axes_value);
    reduce_sum_value.set_attr_keep_dims(false);

    std::vector<Operator> inputs{bmm_x_data};
    std::vector<Operator> outputs{matmul_dw_query, matmul_dw_key, matmul_dw_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradWFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 64, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradW") {
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

TEST_F(attention_qkv_gradw_fusion_test, attention_qkv_gradw_fusion_test_8) {
    // platform unsupported
    ge::Graph graph("attention_qkv_gradw_fusion_test8");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 8;
    opti_compilation_info.soc_version = "soc_version";
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // Batchmatmul_dw
    auto bmm_x_data = op::Data("bmm_x_data");
    TensorDesc bmm_x_desc(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    bmm_x_data.update_input_desc_x(bmm_x_desc);
    bmm_x_data.update_output_desc_y(bmm_x_desc);

    auto kernel_bmm_data = op::Data("kernel_bmm_data");
    TensorDesc kernel_bmm_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_bmm_data.update_input_desc_x(kernel_bmm_desc);
    kernel_bmm_data.update_output_desc_y(kernel_bmm_desc);

    auto bmm = op::BatchMatMulV2("bmm");
    bmm.set_input_x1(bmm_x_data);
    bmm.set_input_x2(kernel_bmm_data);
    bmm.set_attr_adj_x1(false);
    bmm.set_attr_adj_x2(false);
    bmm.set_attr_offset_x(0);

    // conf_transpose
    auto conf_trans_dw_query = op::ConfusionTransposeD("conf_trans_dw_query");
    conf_trans_dw_query.set_input_x(bmm);
    conf_trans_dw_query.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_query.set_attr_shape({12288, 1024});
    conf_trans_dw_query.set_attr_transpose_first(1);
    TensorDesc output_desc_query(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_query.SetOriginShape(ge::Shape({12288, 1024}));
    conf_trans_dw_query.update_output_desc_y(output_desc_query);

    auto conf_trans_dw_key = op::ConfusionTransposeD("conf_trans_dw_key");
    conf_trans_dw_key.set_input_x(bmm);
    conf_trans_dw_key.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_key.set_attr_shape({12288, 1024});
    conf_trans_dw_key.set_attr_transpose_first(1);

    auto conf_trans_dw_value = op::ConfusionTransposeD("conf_trans_dw_value");
    conf_trans_dw_value.set_input_x(bmm);
    conf_trans_dw_value.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_value.set_attr_shape({12288, 1024});
    conf_trans_dw_value.set_attr_transpose_first(1);

    // layer_norm
    auto input_x_data = op::Data("input_x_data");
    TensorDesc input_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_x_data.update_input_desc_x(input_x_desc);
    input_x_data.update_output_desc_y(input_x_desc);

    TensorDesc gamma_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor gamma_tensor(gamma_desc);
    auto gamma = op::Const("gamma").set_attr_value(gamma_tensor);

    TensorDesc beta_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor beta_tensor(beta_desc);
    auto beta = op::Const("beta").set_attr_value(beta_tensor);

    auto layer_norm = op::LayerNorm("layer_norm_0");
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    // matmul_dx_qkv
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    auto matmul_dx_query = op::MatMulV2("matmul_dx_query");
    matmul_dx_query.set_input_x1(conf_trans_dw_query);
    matmul_dx_query.set_input_x2(kernel_query);

    auto matmul_dx_key = op::MatMulV2("matmul_dx_key");
    matmul_dx_key.set_input_x1(conf_trans_dw_key);
    matmul_dx_key.set_input_x2(kernel_key);

    auto matmul_dx_value = op::MatMulV2("matmul_dx_value");
    matmul_dx_value.set_input_x1(conf_trans_dw_value);
    matmul_dx_value.set_input_x2(kernel_value);

    // matmul_dw_qkv
    auto matmul_dw_query = op::MatMulV2("matmul_dw_query");
    matmul_dw_query.set_input_x1(layer_norm, "y");
    matmul_dw_query.set_input_x2(conf_trans_dw_query);
    matmul_dw_query.set_attr_transpose_x1(true);
    matmul_dw_query.set_attr_transpose_x2(false);

    auto matmul_dw_key = op::MatMulV2("matmul_dw_key");
    matmul_dw_key.set_input_x1(layer_norm, "y");
    matmul_dw_key.set_input_x2(conf_trans_dw_key);
    matmul_dw_key.set_attr_transpose_x1(true);
    matmul_dw_key.set_attr_transpose_x2(false);

    auto matmul_dw_value = op::MatMulV2("matmul_dw_value");
    matmul_dw_value.set_input_x1(layer_norm, "y");
    matmul_dw_value.set_input_x2(conf_trans_dw_value);
    matmul_dw_value.set_attr_transpose_x1(true);
    matmul_dw_value.set_attr_transpose_x2(false);

    // reduce_sum_qkv
    std::vector<int64_t> axes_value{1, 2};
    auto reduce_sum_query = op::ReduceSumD("reduce_sum_query");
    reduce_sum_query.set_input_x(conf_trans_dw_query);
    reduce_sum_query.set_attr_axes(axes_value);
    reduce_sum_query.set_attr_keep_dims(false);

    auto reduce_sum_key = op::ReduceSumD("reduce_sum_key");
    reduce_sum_key.set_input_x(conf_trans_dw_key);
    reduce_sum_key.set_attr_axes(axes_value);
    reduce_sum_key.set_attr_keep_dims(false);

    auto reduce_sum_value = op::ReduceSumD("reduce_sum_value");
    reduce_sum_value.set_input_x(conf_trans_dw_value);
    reduce_sum_value.set_attr_axes(axes_value);
    reduce_sum_value.set_attr_keep_dims(false);

    std::vector<Operator> inputs{bmm_x_data, kernel_bmm_data};
    std::vector<Operator> outputs{matmul_dw_query, matmul_dw_key, matmul_dw_value, reduce_sum_query, reduce_sum_key, reduce_sum_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradWFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 64, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradW") {
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

TEST_F(attention_qkv_gradw_fusion_test, attention_qkv_gradw_fusion_test_9) {
    // reduce_sum node match failed
    ge::Graph graph("attention_qkv_gradw_fusion_test1");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // Batchmatmul_dw
    auto bmm_x_data = op::Data("bmm_x_data");
    TensorDesc bmm_x_desc(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    bmm_x_data.update_input_desc_x(bmm_x_desc);
    bmm_x_data.update_output_desc_y(bmm_x_desc);

    auto kernel_bmm_data = op::Data("kernel_bmm_data");
    TensorDesc kernel_bmm_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_bmm_data.update_input_desc_x(kernel_bmm_desc);
    kernel_bmm_data.update_output_desc_y(kernel_bmm_desc);

    auto bmm = op::BatchMatMulV2("bmm");
    bmm.set_input_x1(bmm_x_data);
    bmm.set_input_x2(kernel_bmm_data);
    bmm.set_attr_adj_x1(false);
    bmm.set_attr_adj_x2(false);
    bmm.set_attr_offset_x(0);

    // conf_transpose
    auto conf_trans_dw_query = op::ConfusionTransposeD("conf_trans_dw_query");
    conf_trans_dw_query.set_input_x(bmm);
    conf_trans_dw_query.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_query.set_attr_shape({12288, 1024});
    conf_trans_dw_query.set_attr_transpose_first(1);
    TensorDesc output_desc_query(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_query.SetOriginShape(ge::Shape({12288, 1024}));
    conf_trans_dw_query.update_output_desc_y(output_desc_query);

    auto conf_trans_dw_key = op::ConfusionTransposeD("conf_trans_dw_key");
    conf_trans_dw_key.set_input_x(bmm);
    conf_trans_dw_key.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_key.set_attr_shape({12288, 1024});
    conf_trans_dw_key.set_attr_transpose_first(1);

    auto conf_trans_dw_value = op::ConfusionTransposeD("conf_trans_dw_value");
    conf_trans_dw_value.set_input_x(bmm);
    conf_trans_dw_value.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_value.set_attr_shape({12288, 1024});
    conf_trans_dw_value.set_attr_transpose_first(1);

    // layer_norm
    auto input_x_data = op::Data("input_x_data");
    TensorDesc input_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_x_data.update_input_desc_x(input_x_desc);
    input_x_data.update_output_desc_y(input_x_desc);

    TensorDesc gamma_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor gamma_tensor(gamma_desc);
    auto gamma = op::Const("gamma").set_attr_value(gamma_tensor);

    TensorDesc beta_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor beta_tensor(beta_desc);
    auto beta = op::Const("beta").set_attr_value(beta_tensor);

    auto layer_norm = op::LayerNorm("layer_norm_0");
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    // matmul_dx_qkv
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    auto matmul_dx_query = op::MatMulV2("matmul_dx_query");
    matmul_dx_query.set_input_x1(conf_trans_dw_query);
    matmul_dx_query.set_input_x2(kernel_query);

    auto matmul_dx_key = op::MatMulV2("matmul_dx_key");
    matmul_dx_key.set_input_x1(conf_trans_dw_key);
    matmul_dx_key.set_input_x2(kernel_key);

    auto matmul_dx_value = op::MatMulV2("matmul_dx_value");
    matmul_dx_value.set_input_x1(conf_trans_dw_value);
    matmul_dx_value.set_input_x2(kernel_value);

    // matmul_dw_qkv
    auto matmul_dw_query = op::MatMulV2("matmul_dw_query");
    matmul_dw_query.set_input_x1(layer_norm, "y");
    matmul_dw_query.set_input_x2(conf_trans_dw_query);
    matmul_dw_query.set_attr_transpose_x1(true);
    matmul_dw_query.set_attr_transpose_x2(false);
    TensorDesc output_desc_dw_query(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_dw_query.update_output_desc_y(output_desc_dw_query);

    auto matmul_dw_key = op::MatMulV2("matmul_dw_key");
    matmul_dw_key.set_input_x1(layer_norm, "y");
    matmul_dw_key.set_input_x2(conf_trans_dw_key);
    matmul_dw_key.set_attr_transpose_x1(true);
    matmul_dw_key.set_attr_transpose_x2(false);
    TensorDesc output_desc_dw_key(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_dw_key.update_output_desc_y(output_desc_dw_key);

    auto matmul_dw_value = op::MatMulV2("matmul_dw_value");
    matmul_dw_value.set_input_x1(layer_norm, "y");
    matmul_dw_value.set_input_x2(conf_trans_dw_value);
    matmul_dw_value.set_attr_transpose_x1(true);
    matmul_dw_value.set_attr_transpose_x2(false);
    TensorDesc output_desc_dw_value(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_dw_value.update_output_desc_y(output_desc_dw_value);

    // reduce_sum_qkv
    std::vector<int64_t> axes_value{1, 2};
    auto reduce_sum_query = op::ReduceSumD("reduce_sum_query");
    reduce_sum_query.set_input_x(conf_trans_dw_query);
    reduce_sum_query.set_attr_axes(axes_value);
    reduce_sum_query.set_attr_keep_dims(false);

    auto reduce_sum_key = op::ReduceSumD("reduce_sum_key");
    reduce_sum_key.set_input_x(conf_trans_dw_key);
    reduce_sum_key.set_attr_axes(axes_value);
    reduce_sum_key.set_attr_keep_dims(false);

    TensorDesc desc_input_size_1(ge::Shape({1}), FORMAT_ND, DT_INT32);
    Tensor axis_tensor(desc_input_size_1);
    uint32_t *axis_tensor_value = new uint32_t[1]{2};
    axis_tensor.SetData((uint8_t *) axis_tensor_value, 1 * sizeof(uint32_t));
    auto begin = op::Constant().set_attr_value(axis_tensor);
    delete []axis_tensor_value;
    auto reduce_sum_value = op::ReduceSum("reduce_sum_value");
    reduce_sum_value.set_input_x(conf_trans_dw_value);
    reduce_sum_value.set_input_axes(begin);
    reduce_sum_value.set_attr_keep_dims(false);

    std::vector<Operator> inputs{bmm_x_data, kernel_bmm_data};
    std::vector<Operator> outputs{matmul_dw_query, matmul_dw_key, matmul_dw_value, reduce_sum_query, reduce_sum_key, reduce_sum_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradWFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 64, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradW") {
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

TEST_F(attention_qkv_gradw_fusion_test, attention_qkv_gradw_fusion_test_10) {
    // uusupported dtype
    ge::Graph graph("attention_qkv_gradw_fusion_test10");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // Batchmatmul_dw
    auto bmm_x_data = op::Data("bmm_x_data");
    TensorDesc bmm_x_desc(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    bmm_x_data.update_input_desc_x(bmm_x_desc);
    bmm_x_data.update_output_desc_y(bmm_x_desc);

    auto kernel_bmm_data = op::Data("kernel_bmm_data");
    TensorDesc kernel_bmm_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_bmm_data.update_input_desc_x(kernel_bmm_desc);
    kernel_bmm_data.update_output_desc_y(kernel_bmm_desc);

    auto bmm = op::BatchMatMulV2("bmm");
    bmm.set_input_x1(bmm_x_data);
    bmm.set_input_x2(kernel_bmm_data);
    bmm.set_attr_adj_x1(false);
    bmm.set_attr_adj_x2(false);
    bmm.set_attr_offset_x(0);

    // conf_transpose
    auto conf_trans_dw_query = op::ConfusionTransposeD("conf_trans_dw_query");
    conf_trans_dw_query.set_input_x(bmm);
    conf_trans_dw_query.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_query.set_attr_shape({12288, 1024});
    conf_trans_dw_query.set_attr_transpose_first(1);
    TensorDesc output_desc_query(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT);
    output_desc_query.SetOriginShape(ge::Shape({12288, 1024}));
    conf_trans_dw_query.update_output_desc_y(output_desc_query);

    auto conf_trans_dw_key = op::ConfusionTransposeD("conf_trans_dw_key");
    conf_trans_dw_key.set_input_x(bmm);
    conf_trans_dw_key.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_key.set_attr_shape({12288, 1024});
    conf_trans_dw_key.set_attr_transpose_first(1);

    auto conf_trans_dw_value = op::ConfusionTransposeD("conf_trans_dw_value");
    conf_trans_dw_value.set_input_x(bmm);
    conf_trans_dw_value.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_value.set_attr_shape({12288, 1024});
    conf_trans_dw_value.set_attr_transpose_first(1);

    // layer_norm
    auto input_x_data = op::Data("input_x_data");
    TensorDesc input_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_x_data.update_input_desc_x(input_x_desc);
    input_x_data.update_output_desc_y(input_x_desc);

    TensorDesc gamma_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor gamma_tensor(gamma_desc);
    auto gamma = op::Const("gamma").set_attr_value(gamma_tensor);

    TensorDesc beta_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor beta_tensor(beta_desc);
    auto beta = op::Const("beta").set_attr_value(beta_tensor);

    auto layer_norm = op::LayerNorm("layer_norm_0");
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    // matmul_dx_qkv
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    auto matmul_dx_query = op::MatMulV2("matmul_dx_query");
    matmul_dx_query.set_input_x1(conf_trans_dw_query);
    matmul_dx_query.set_input_x2(kernel_query);

    auto matmul_dx_key = op::MatMulV2("matmul_dx_key");
    matmul_dx_key.set_input_x1(conf_trans_dw_key);
    matmul_dx_key.set_input_x2(kernel_key);

    auto matmul_dx_value = op::MatMulV2("matmul_dx_value");
    matmul_dx_value.set_input_x1(conf_trans_dw_value);
    matmul_dx_value.set_input_x2(kernel_value);

    // matmul_dw_qkv
    auto matmul_dw_query = op::MatMulV2("matmul_dw_query");
    matmul_dw_query.set_input_x1(layer_norm, "y");
    matmul_dw_query.set_input_x2(conf_trans_dw_query);
    matmul_dw_query.set_attr_transpose_x1(true);
    matmul_dw_query.set_attr_transpose_x2(false);
    TensorDesc output_desc_dw_query(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_dw_query.update_output_desc_y(output_desc_dw_query);

    auto matmul_dw_key = op::MatMulV2("matmul_dw_key");
    matmul_dw_key.set_input_x1(layer_norm, "y");
    matmul_dw_key.set_input_x2(conf_trans_dw_key);
    matmul_dw_key.set_attr_transpose_x1(true);
    matmul_dw_key.set_attr_transpose_x2(false);
    TensorDesc output_desc_dw_key(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_dw_key.update_output_desc_y(output_desc_dw_key);

    auto matmul_dw_value = op::MatMulV2("matmul_dw_value");
    matmul_dw_value.set_input_x1(layer_norm, "y");
    matmul_dw_value.set_input_x2(conf_trans_dw_value);
    matmul_dw_value.set_attr_transpose_x1(true);
    matmul_dw_value.set_attr_transpose_x2(false);
    TensorDesc output_desc_dw_value(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_dw_value.update_output_desc_y(output_desc_dw_value);

    // reduce_sum_qkv
    std::vector<int64_t> axes_value{1, 2};
    auto reduce_sum_query = op::ReduceSumD("reduce_sum_query");
    reduce_sum_query.set_input_x(conf_trans_dw_query);
    reduce_sum_query.set_attr_axes(axes_value);
    reduce_sum_query.set_attr_keep_dims(false);

    auto reduce_sum_key = op::ReduceSumD("reduce_sum_key");
    reduce_sum_key.set_input_x(conf_trans_dw_key);
    reduce_sum_key.set_attr_axes(axes_value);
    reduce_sum_key.set_attr_keep_dims(false);

    auto reduce_sum_value = op::ReduceSumD("reduce_sum_value");
    reduce_sum_value.set_input_x(conf_trans_dw_value);
    reduce_sum_value.set_attr_axes(axes_value);
    reduce_sum_value.set_attr_keep_dims(false);

    std::vector<Operator> inputs{bmm_x_data, kernel_bmm_data};
    std::vector<Operator> outputs{matmul_dw_query, matmul_dw_key, matmul_dw_value, reduce_sum_query, reduce_sum_key, reduce_sum_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradWFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 64, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradW") {
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

TEST_F(attention_qkv_gradw_fusion_test, attention_qkv_gradw_fusion_test_11) {
    // unsupported format
    ge::Graph graph("attention_qkv_gradw_fusion_test11");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "soc_version";
    platform_info.ai_core_intrinsic_dtype_map["Intrinsic_scatter_vconv"] = {"deq", "f162s32a"};
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    // Batchmatmul_dw
    auto bmm_x_data = op::Data("bmm_x_data");
    TensorDesc bmm_x_desc(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    bmm_x_data.update_input_desc_x(bmm_x_desc);
    bmm_x_data.update_output_desc_y(bmm_x_desc);

    auto kernel_bmm_data = op::Data("kernel_bmm_data");
    TensorDesc kernel_bmm_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    kernel_bmm_data.update_input_desc_x(kernel_bmm_desc);
    kernel_bmm_data.update_output_desc_y(kernel_bmm_desc);

    auto bmm = op::BatchMatMulV2("bmm");
    bmm.set_input_x1(bmm_x_data);
    bmm.set_input_x2(kernel_bmm_data);
    bmm.set_attr_adj_x1(false);
    bmm.set_attr_adj_x2(false);
    bmm.set_attr_offset_x(0);

    // conf_transpose
    auto conf_trans_dw_query = op::ConfusionTransposeD("conf_trans_dw_query");
    conf_trans_dw_query.set_input_x(bmm);
    conf_trans_dw_query.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_query.set_attr_shape({12288, 1024});
    conf_trans_dw_query.set_attr_transpose_first(1);
    TensorDesc output_desc_query(ge::Shape({64, 768, 16, 16}), FORMAT_NHWC, DT_FLOAT16);
    output_desc_query.SetOriginShape(ge::Shape({12288, 1024}));
    conf_trans_dw_query.update_output_desc_y(output_desc_query);

    auto conf_trans_dw_key = op::ConfusionTransposeD("conf_trans_dw_key");
    conf_trans_dw_key.set_input_x(bmm);
    conf_trans_dw_key.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_key.set_attr_shape({12288, 1024});
    conf_trans_dw_key.set_attr_transpose_first(1);

    auto conf_trans_dw_value = op::ConfusionTransposeD("conf_trans_dw_value");
    conf_trans_dw_value.set_input_x(bmm);
    conf_trans_dw_value.set_attr_perm({0, 2, 1, 3});
    conf_trans_dw_value.set_attr_shape({12288, 1024});
    conf_trans_dw_value.set_attr_transpose_first(1);

    // layer_norm
    auto input_x_data = op::Data("input_x_data");
    TensorDesc input_x_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_x_data.update_input_desc_x(input_x_desc);
    input_x_data.update_output_desc_y(input_x_desc);

    TensorDesc gamma_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor gamma_tensor(gamma_desc);
    auto gamma = op::Const("gamma").set_attr_value(gamma_tensor);

    TensorDesc beta_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor beta_tensor(beta_desc);
    auto beta = op::Const("beta").set_attr_value(beta_tensor);

    auto layer_norm = op::LayerNorm("layer_norm_0");
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    // matmul_dx_qkv
    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    auto matmul_dx_query = op::MatMulV2("matmul_dx_query");
    matmul_dx_query.set_input_x1(conf_trans_dw_query);
    matmul_dx_query.set_input_x2(kernel_query);

    auto matmul_dx_key = op::MatMulV2("matmul_dx_key");
    matmul_dx_key.set_input_x1(conf_trans_dw_key);
    matmul_dx_key.set_input_x2(kernel_key);

    auto matmul_dx_value = op::MatMulV2("matmul_dx_value");
    matmul_dx_value.set_input_x1(conf_trans_dw_value);
    matmul_dx_value.set_input_x2(kernel_value);

    // matmul_dw_qkv
    auto matmul_dw_query = op::MatMulV2("matmul_dw_query");
    matmul_dw_query.set_input_x1(layer_norm, "y");
    matmul_dw_query.set_input_x2(conf_trans_dw_query);
    matmul_dw_query.set_attr_transpose_x1(true);
    matmul_dw_query.set_attr_transpose_x2(false);
    TensorDesc output_desc_dw_query(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_dw_query.update_output_desc_y(output_desc_dw_query);

    auto matmul_dw_key = op::MatMulV2("matmul_dw_key");
    matmul_dw_key.set_input_x1(layer_norm, "y");
    matmul_dw_key.set_input_x2(conf_trans_dw_key);
    matmul_dw_key.set_attr_transpose_x1(true);
    matmul_dw_key.set_attr_transpose_x2(false);
    TensorDesc output_desc_dw_key(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_dw_key.update_output_desc_y(output_desc_dw_key);

    auto matmul_dw_value = op::MatMulV2("matmul_dw_value");
    matmul_dw_value.set_input_x1(layer_norm, "y");
    matmul_dw_value.set_input_x2(conf_trans_dw_value);
    matmul_dw_value.set_attr_transpose_x1(true);
    matmul_dw_value.set_attr_transpose_x2(false);
    TensorDesc output_desc_dw_value(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_dw_value.update_output_desc_y(output_desc_dw_value);

    // reduce_sum_qkv
    std::vector<int64_t> axes_value{1, 2};
    auto reduce_sum_query = op::ReduceSumD("reduce_sum_query");
    reduce_sum_query.set_input_x(conf_trans_dw_query);
    reduce_sum_query.set_attr_axes(axes_value);
    reduce_sum_query.set_attr_keep_dims(false);

    auto reduce_sum_key = op::ReduceSumD("reduce_sum_key");
    reduce_sum_key.set_input_x(conf_trans_dw_key);
    reduce_sum_key.set_attr_axes(axes_value);
    reduce_sum_key.set_attr_keep_dims(false);

    auto reduce_sum_value = op::ReduceSumD("reduce_sum_value");
    reduce_sum_value.set_input_x(conf_trans_dw_value);
    reduce_sum_value.set_attr_axes(axes_value);
    reduce_sum_value.set_attr_keep_dims(false);

    std::vector<Operator> inputs{bmm_x_data, kernel_bmm_data};
    std::vector<Operator> outputs{matmul_dw_query, matmul_dw_key, matmul_dw_value, reduce_sum_query, reduce_sum_key, reduce_sum_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionQKVGradWFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{64, 64, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionQKVGradW") {
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