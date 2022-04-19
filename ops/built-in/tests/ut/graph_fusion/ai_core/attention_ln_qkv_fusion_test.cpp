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

class attention_ln_qkv_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "attention_ln_qkv_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "attention_ln_qkv_fusion_test TearDown" << std::endl;
  }
};

TEST_F(attention_ln_qkv_fusion_test, attention_ln_qkv_fusion_test_1) {
    ge::Graph graph("attention_ln_qkv_fusion_test1");

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 8;
    opti_compilation_info.soc_version = "Ascend710";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

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
    TensorDesc input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc input_desc_gamma(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc input_desc_beta(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    TensorDesc output_desc_mean(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_variance(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    layer_norm.update_input_desc_x(input_desc_x);
    layer_norm.update_input_desc_gamma(input_desc_gamma);
    layer_norm.update_input_desc_beta(input_desc_beta);
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.update_output_desc_mean(output_desc_mean);
    layer_norm.update_output_desc_variance(output_desc_variance);
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    auto end_op_mean = op::Cast("end_op_mean");
    end_op_mean.set_input_x(layer_norm, "mean");
    end_op_mean.set_attr_dst_type(0);

    auto end_op_variance = op::Cast("end_op_variance");
    end_op_variance.set_input_x(layer_norm, "variance");
    end_op_variance.set_attr_dst_type(1);

    auto trans_data_0 = op::TransData("trans_data_0");
    TensorDesc trans_data_in_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc trans_data_out_desc(ge::Shape({12288, 1024}), FORMAT_ND, DT_FLOAT16);
    trans_data_0.update_input_desc_src(trans_data_in_desc);
    trans_data_0.update_output_desc_dst(trans_data_out_desc);
    trans_data_0.set_input_src(layer_norm, "y");
    trans_data_0.set_attr_src_format("FRACTAL_NZ");
    trans_data_0.set_attr_dst_format("ND");

    auto reformat_0 = op::ReFormat("reformat");
    TensorDesc reformat_in_desc(ge::Shape({12288, 1024}), FORMAT_ND, DT_FLOAT16);
    TensorDesc reformat_out_desc(ge::Shape({12288, 1024}), FORMAT_NHWC, DT_FLOAT16);
    reformat_0.update_input_desc_x(reformat_in_desc);
    reformat_0.update_output_desc_y(reformat_out_desc);
    reformat_0.set_input_x(trans_data_0);

    auto trans_data_1 = op::TransData("trans_data_1");
    TensorDesc trans_data_in_desc1(ge::Shape({12288, 1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc trans_data_out_desc1(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    trans_data_1.update_input_desc_src(trans_data_in_desc1);
    trans_data_1.update_output_desc_dst(trans_data_out_desc1);
    trans_data_1.set_input_src(reformat_0);
    trans_data_1.set_attr_src_format("NHWC");
    trans_data_1.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_2 = op::TransData("trans_data_2");
    TensorDesc trans_data_in_desc2(ge::Shape({12288, 1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc trans_data_out_desc2(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    trans_data_2.update_input_desc_src(trans_data_in_desc2);
    trans_data_2.update_output_desc_dst(trans_data_out_desc2);
    trans_data_2.set_input_src(reformat_0);
    trans_data_2.set_attr_src_format("NHWC");
    trans_data_2.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_3 = op::TransData("trans_data_3");
    TensorDesc trans_data_in_desc3(ge::Shape({12288, 1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc trans_data_out_desc3(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    trans_data_3.update_input_desc_src(trans_data_in_desc3);
    trans_data_3.update_output_desc_dst(trans_data_out_desc3);
    trans_data_3.set_input_src(reformat_0);
    trans_data_3.set_attr_src_format("NHWC");
    trans_data_3.set_attr_dst_format("FRACTAL_NZ");

    TensorDesc add_desc_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor_1(add_desc_1);
    auto add_const_op_1 = op::Const("add_const_op_1").set_attr_value(add_tensor_1);
    auto add_1 = op::Add("add_1");
    TensorDesc add_input_desc_x1_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc add_input_desc_x2_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc add_output_desc_y_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    add_1.update_input_desc_x1(add_input_desc_x1_1);
    add_1.update_input_desc_x2(add_input_desc_x2_1);
    add_1.update_output_desc_y(add_output_desc_y_1);
    add_1.set_input_x1(reformat_0);
    add_1.set_input_x2(add_const_op_1);

    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    TensorDesc bias_query_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_query_tensor(bias_query_desc);
    auto bias_query = op::Const("bias_query").set_attr_value(bias_query_tensor);
    TensorDesc bias_key_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_key_tensor(bias_key_desc);
    auto bias_key = op::Const("bias_key").set_attr_value(bias_key_tensor);
    TensorDesc bias_value_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_value_tensor(bias_value_desc);
    auto bias_value = op::Const("bias_value").set_attr_value(bias_value_tensor);

    auto matmul_query = op::MatMulV2("matmul_query");
    TensorDesc mm_q_input_desc_x1(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_q_input_desc_x2(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_q_input_desc_bias(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    TensorDesc mm_q_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_query.update_input_desc_x1(mm_q_input_desc_x1);
    matmul_query.update_input_desc_x2(mm_q_input_desc_x2);
    matmul_query.update_input_desc_bias(mm_q_input_desc_bias);
    matmul_query.update_output_desc_y(mm_q_output_desc_y);
    matmul_query.set_input_x1(trans_data_1);
    matmul_query.set_input_x2(kernel_query);
    matmul_query.set_input_bias(bias_query);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(false);

    auto matmul_key = op::MatMulV2("matmul_key");
    TensorDesc mm_k_input_desc_x1(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_k_input_desc_x2(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_k_input_desc_bias(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    TensorDesc mm_k_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_key.update_input_desc_x1(mm_k_input_desc_x1);
    matmul_key.update_input_desc_x2(mm_k_input_desc_x2);
    matmul_key.update_input_desc_bias(mm_k_input_desc_bias);
    matmul_key.update_output_desc_y(mm_k_output_desc_y);
    matmul_key.set_input_x1(trans_data_2);
    matmul_key.set_input_x2(kernel_key);
    matmul_key.set_input_bias(bias_key);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(false);

    auto matmul_value = op::MatMulV2("matmul_value");
    TensorDesc mm_v_input_desc_x1(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_v_input_desc_x2(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_v_input_desc_bias(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    TensorDesc mm_v_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_value.update_input_desc_x1(mm_v_input_desc_x1);
    matmul_value.update_input_desc_x2(mm_v_input_desc_x2);
    matmul_value.update_input_desc_bias(mm_v_input_desc_bias);
    matmul_value.update_output_desc_y(mm_v_output_desc_y);
    matmul_value.set_input_x1(trans_data_3);
    matmul_value.set_input_x2(kernel_value);
    matmul_value.set_input_bias(bias_value);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(false);

    auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
    TensorDesc conf_trans_query_input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc conf_trans_query_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    conf_transpose_query.update_input_desc_x(conf_trans_query_input_desc_x);
    conf_transpose_query.update_output_desc_y(conf_trans_query_output_desc_y);
    conf_transpose_query.set_input_x(matmul_query);
    conf_transpose_query.set_attr_perm({0, 2, 1, 3});
    conf_transpose_query.set_attr_shape({24, 512, 16, 64});
    conf_transpose_query.set_attr_transpose_first(1);

    auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
    TensorDesc conf_trans_key_input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc conf_trans_key_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    conf_transpose_key.update_input_desc_x(conf_trans_key_input_desc_x);
    conf_transpose_key.update_output_desc_y(conf_trans_key_output_desc_y);
    conf_transpose_key.set_input_x(matmul_key);
    conf_transpose_key.set_attr_perm({0, 2, 1, 3});
    conf_transpose_key.set_attr_shape({24, 512, 16, 64});
    conf_transpose_key.set_attr_transpose_first(1);

    auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
    TensorDesc conf_trans_value_input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc conf_trans_value_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    conf_transpose_value.update_input_desc_x(conf_trans_value_input_desc_x);
    conf_transpose_value.update_output_desc_y(conf_trans_value_output_desc_y);
    conf_transpose_value.set_input_x(matmul_value);
    conf_transpose_value.set_attr_perm({0, 2, 1, 3});
    conf_transpose_value.set_attr_shape({24, 512, 16, 64});
    conf_transpose_value.set_attr_transpose_first(1);

    TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor(add_desc);
    auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
    auto add_0 = op::Add("add_0");
    TensorDesc add_input_desc_x1(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc add_input_desc_x2(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc add_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    add_0.update_input_desc_x1(add_input_desc_x1);
    add_0.update_input_desc_x2(add_input_desc_x2);
    add_0.update_output_desc_y(add_output_desc_y);
    add_0.set_input_x1(layer_norm, "y");
    add_0.set_input_x2(add_const_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{24, 16, 4, 32, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionLnQKV") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(1);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, true);
    EXPECT_EQ(shapeMatch, true);
}

TEST_F(attention_ln_qkv_fusion_test, attention_ln_qkv_fusion_test_2) {
    // origin_shape not aligned
    ge::Graph graph("attention_ln_qkv_fusion_test2");

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 8;
    opti_compilation_info.soc_version = "Ascend710";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

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
    TensorDesc input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc input_desc_gamma(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc input_desc_beta(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1020}));
    TensorDesc output_desc_mean(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_variance(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    layer_norm.update_input_desc_x(input_desc_x);
    layer_norm.update_input_desc_gamma(input_desc_gamma);
    layer_norm.update_input_desc_beta(input_desc_beta);
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.update_output_desc_mean(output_desc_mean);
    layer_norm.update_output_desc_variance(output_desc_variance);
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    auto end_op_mean = op::Cast("end_op_mean");
    end_op_mean.set_input_x(layer_norm, "mean");
    end_op_mean.set_attr_dst_type(0);

    auto end_op_variance = op::Cast("end_op_variance");
    end_op_variance.set_input_x(layer_norm, "variance");
    end_op_variance.set_attr_dst_type(1);

    auto trans_data_0 = op::TransData("trans_data_0");
    trans_data_0.set_input_src(layer_norm, "y");
    trans_data_0.set_attr_src_format("FRACTAL_NZ");
    trans_data_0.set_attr_dst_format("ND");

    auto reformat_0 = op::ReFormat("reformat");
    reformat_0.set_input_x(trans_data_0);

    auto trans_data_1 = op::TransData("trans_data_1");
    trans_data_1.set_input_src(reformat_0);
    trans_data_1.set_attr_src_format("NHWC");
    trans_data_1.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_2 = op::TransData("trans_data_2");
    trans_data_2.set_input_src(reformat_0);
    trans_data_2.set_attr_src_format("NHWC");
    trans_data_2.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_3 = op::TransData("trans_data_3");
    trans_data_3.set_input_src(reformat_0);
    trans_data_3.set_attr_src_format("NHWC");
    trans_data_3.set_attr_dst_format("FRACTAL_NZ");

    TensorDesc add_desc_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor_1(add_desc_1);
    auto add_const_op_1 = op::Const("add_const_op_1").set_attr_value(add_tensor_1);
    auto add_1 = op::Add("add_1");
    add_1.set_input_x1(reformat_0);
    add_1.set_input_x2(add_const_op_1);

    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    TensorDesc bias_query_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_query_tensor(bias_query_desc);
    auto bias_query = op::Const("bias_query").set_attr_value(bias_query_tensor);
    TensorDesc bias_key_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_key_tensor(bias_key_desc);
    auto bias_key = op::Const("bias_key").set_attr_value(bias_key_tensor);
    TensorDesc bias_value_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_value_tensor(bias_value_desc);
    auto bias_value = op::Const("bias_value").set_attr_value(bias_value_tensor);

    auto matmul_query = op::MatMulV2("matmul_query");
    matmul_query.set_input_x1(trans_data_1);
    matmul_query.set_input_x2(kernel_query);
    matmul_query.set_input_bias(bias_query);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(false);

    auto matmul_key = op::MatMulV2("matmul_key");
    matmul_key.set_input_x1(trans_data_2);
    matmul_key.set_input_x2(kernel_key);
    matmul_key.set_input_bias(bias_key);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(false);

    auto matmul_value = op::MatMulV2("matmul_value");
    matmul_value.set_input_x1(trans_data_3);
    matmul_value.set_input_x2(kernel_value);
    matmul_value.set_input_bias(bias_value);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(false);

    auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
    conf_transpose_query.set_input_x(matmul_query);
    conf_transpose_query.set_attr_perm({0, 2, 1, 3});
    conf_transpose_query.set_attr_shape({24, 512, 16, 64});
    conf_transpose_query.set_attr_transpose_first(1);

    auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
    conf_transpose_key.set_input_x(matmul_key);
    conf_transpose_key.set_attr_perm({0, 2, 1, 3});
    conf_transpose_key.set_attr_shape({24, 512, 16, 64});
    conf_transpose_key.set_attr_transpose_first(1);

    auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
    conf_transpose_value.set_input_x(matmul_value);
    conf_transpose_value.set_attr_perm({0, 2, 1, 3});
    conf_transpose_value.set_attr_shape({24, 512, 16, 64});
    conf_transpose_value.set_attr_transpose_first(1);

    TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor(add_desc);
    auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
    auto add_0 = op::Add("add_0");
    add_0.set_input_x1(layer_norm, "y");
    add_0.set_input_x2(add_const_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{24, 16, 4, 32, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionLnQKV") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(1);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}

TEST_F(attention_ln_qkv_fusion_test, attention_ln_qkv_fusion_test_3) {
    // confusion_transpose out_shape check failed
    ge::Graph graph("attention_ln_qkv_fusion_test3");

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 8;
    opti_compilation_info.soc_version = "Ascend710";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

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
    TensorDesc input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc input_desc_gamma(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc input_desc_beta(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    TensorDesc output_desc_mean(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_variance(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    layer_norm.update_input_desc_x(input_desc_x);
    layer_norm.update_input_desc_gamma(input_desc_gamma);
    layer_norm.update_input_desc_beta(input_desc_beta);
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.update_output_desc_mean(output_desc_mean);
    layer_norm.update_output_desc_variance(output_desc_variance);
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    auto end_op_mean = op::Cast("end_op_mean");
    end_op_mean.set_input_x(layer_norm, "mean");
    end_op_mean.set_attr_dst_type(0);

    auto end_op_variance = op::Cast("end_op_variance");
    end_op_variance.set_input_x(layer_norm, "variance");
    end_op_variance.set_attr_dst_type(1);

    auto trans_data_0 = op::TransData("trans_data_0");
    trans_data_0.set_input_src(layer_norm, "y");
    trans_data_0.set_attr_src_format("FRACTAL_NZ");
    trans_data_0.set_attr_dst_format("ND");

    auto reformat_0 = op::ReFormat("reformat");
    reformat_0.set_input_x(trans_data_0);

    auto trans_data_1 = op::TransData("trans_data_1");
    trans_data_1.set_input_src(reformat_0);
    trans_data_1.set_attr_src_format("NHWC");
    trans_data_1.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_2 = op::TransData("trans_data_2");
    trans_data_2.set_input_src(reformat_0);
    trans_data_2.set_attr_src_format("NHWC");
    trans_data_2.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_3 = op::TransData("trans_data_3");
    trans_data_3.set_input_src(reformat_0);
    trans_data_3.set_attr_src_format("NHWC");
    trans_data_3.set_attr_dst_format("FRACTAL_NZ");

    TensorDesc add_desc_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor_1(add_desc_1);
    auto add_const_op_1 = op::Const("add_const_op_1").set_attr_value(add_tensor_1);
    auto add_1 = op::Add("add_1");
    add_1.set_input_x1(reformat_0);
    add_1.set_input_x2(add_const_op_1);

    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    TensorDesc bias_query_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_query_tensor(bias_query_desc);
    auto bias_query = op::Const("bias_query").set_attr_value(bias_query_tensor);
    TensorDesc bias_key_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_key_tensor(bias_key_desc);
    auto bias_key = op::Const("bias_key").set_attr_value(bias_key_tensor);
    TensorDesc bias_value_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_value_tensor(bias_value_desc);
    auto bias_value = op::Const("bias_value").set_attr_value(bias_value_tensor);

    auto matmul_query = op::MatMulV2("matmul_query");
    matmul_query.set_input_x1(trans_data_1);
    matmul_query.set_input_x2(kernel_query);
    matmul_query.set_input_bias(bias_query);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(false);

    auto matmul_key = op::MatMulV2("matmul_key");
    matmul_key.set_input_x1(trans_data_2);
    matmul_key.set_input_x2(kernel_key);
    matmul_key.set_input_bias(bias_key);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(false);

    auto matmul_value = op::MatMulV2("matmul_value");
    matmul_value.set_input_x1(trans_data_3);
    matmul_value.set_input_x2(kernel_value);
    matmul_value.set_input_bias(bias_value);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(false);

    auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
    TensorDesc conf_trans_query_input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc conf_trans_query_output_desc_y(ge::Shape({24, 32, 16, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    conf_transpose_query.update_input_desc_x(conf_trans_query_input_desc_x);
    conf_transpose_query.update_output_desc_y(conf_trans_query_output_desc_y);
    conf_transpose_query.set_input_x(matmul_query);
    conf_transpose_query.set_attr_perm({0, 2, 1, 3});
    conf_transpose_query.set_attr_shape({24, 512, 16, 64});
    conf_transpose_query.set_attr_transpose_first(1);

    auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
    TensorDesc conf_trans_key_input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc conf_trans_key_output_desc_y(ge::Shape({24, 32, 16, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    conf_transpose_key.update_input_desc_x(conf_trans_key_input_desc_x);
    conf_transpose_key.update_output_desc_y(conf_trans_key_output_desc_y);
    conf_transpose_key.set_input_x(matmul_key);
    conf_transpose_key.set_attr_perm({0, 2, 1, 3});
    conf_transpose_key.set_attr_shape({24, 512, 16, 64});
    conf_transpose_key.set_attr_transpose_first(1);

    auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
    TensorDesc conf_trans_value_input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc conf_trans_value_output_desc_y(ge::Shape({24, 32, 16, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    conf_transpose_value.update_input_desc_x(conf_trans_value_input_desc_x);
    conf_transpose_value.update_output_desc_y(conf_trans_value_output_desc_y);
    conf_transpose_value.set_input_x(matmul_value);
    conf_transpose_value.set_attr_perm({0, 2, 1, 3});
    conf_transpose_value.set_attr_shape({24, 512, 16, 64});
    conf_transpose_value.set_attr_transpose_first(1);

    TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor(add_desc);
    auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
    auto add_0 = op::Add("add_0");
    add_0.set_input_x1(layer_norm, "y");
    add_0.set_input_x2(add_const_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{24, 16, 4, 32, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionLnQKV") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(1);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}

TEST_F(attention_ln_qkv_fusion_test, attention_ln_qkv_fusion_test_4) {
    // trans_data->matmulv2 pattern check failed
    ge::Graph graph("attention_ln_qkv_fusion_test4");

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 8;
    opti_compilation_info.soc_version = "Ascend710";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

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
    TensorDesc input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc input_desc_gamma(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc input_desc_beta(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    TensorDesc output_desc_mean(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_variance(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    layer_norm.update_input_desc_x(input_desc_x);
    layer_norm.update_input_desc_gamma(input_desc_gamma);
    layer_norm.update_input_desc_beta(input_desc_beta);
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.update_output_desc_mean(output_desc_mean);
    layer_norm.update_output_desc_variance(output_desc_variance);
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    auto end_op_mean = op::Cast("end_op_mean");
    end_op_mean.set_input_x(layer_norm, "mean");
    end_op_mean.set_attr_dst_type(0);

    auto end_op_variance = op::Cast("end_op_variance");
    end_op_variance.set_input_x(layer_norm, "variance");
    end_op_variance.set_attr_dst_type(1);

    auto trans_data_0 = op::TransData("trans_data_0");
    trans_data_0.set_input_src(layer_norm, "y");
    trans_data_0.set_attr_src_format("FRACTAL_NZ");
    trans_data_0.set_attr_dst_format("ND");

    auto reformat_0 = op::ReFormat("reformat");
    reformat_0.set_input_x(trans_data_0);

    auto trans_data_1 = op::TransData("trans_data_1");
    trans_data_1.set_input_src(reformat_0);
    trans_data_1.set_attr_src_format("NHWC");
    trans_data_1.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_2 = op::TransData("trans_data_2");
    trans_data_2.set_input_src(reformat_0);
    trans_data_2.set_attr_src_format("NHWC");
    trans_data_2.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_3 = op::TransData("trans_data_3");
    trans_data_3.set_input_src(reformat_0);
    trans_data_3.set_attr_src_format("NHWC");
    trans_data_3.set_attr_dst_format("FRACTAL_NZ");

    TensorDesc add_desc_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor_1(add_desc_1);
    auto add_const_op_1 = op::Const("add_const_op_1").set_attr_value(add_tensor_1);
    auto add_1 = op::Add("add_1");
    add_1.set_input_x1(reformat_0);
    add_1.set_input_x2(add_const_op_1);

    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    TensorDesc bias_query_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_query_tensor(bias_query_desc);
    auto bias_query = op::Const("bias_query").set_attr_value(bias_query_tensor);
    TensorDesc bias_key_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_key_tensor(bias_key_desc);
    auto bias_key = op::Const("bias_key").set_attr_value(bias_key_tensor);
    TensorDesc bias_value_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_value_tensor(bias_value_desc);
    auto bias_value = op::Const("bias_value").set_attr_value(bias_value_tensor);

    auto matmul_query = op::MatMul("matmul_query");
    matmul_query.set_input_x1(trans_data_1);
    matmul_query.set_input_x2(kernel_query);
    matmul_query.set_input_bias(bias_query);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(false);

    auto matmul_key = op::MatMulV2("matmul_key");
    matmul_key.set_input_x1(trans_data_2);
    matmul_key.set_input_x2(kernel_key);
    matmul_key.set_input_bias(bias_key);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(false);

    auto matmul_value = op::MatMulV2("matmul_value");
    matmul_value.set_input_x1(trans_data_3);
    matmul_value.set_input_x2(kernel_value);
    matmul_value.set_input_bias(bias_value);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(false);

    auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
    conf_transpose_query.set_input_x(matmul_query);
    conf_transpose_query.set_attr_perm({0, 2, 1, 3});
    conf_transpose_query.set_attr_shape({24, 512, 16, 64});
    conf_transpose_query.set_attr_transpose_first(1);

    auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
    conf_transpose_key.set_input_x(matmul_key);
    conf_transpose_key.set_attr_perm({0, 2, 1, 3});
    conf_transpose_key.set_attr_shape({24, 512, 16, 64});
    conf_transpose_key.set_attr_transpose_first(1);

    auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
    conf_transpose_value.set_input_x(matmul_value);
    conf_transpose_value.set_attr_perm({0, 2, 1, 3});
    conf_transpose_value.set_attr_shape({24, 512, 16, 64});
    conf_transpose_value.set_attr_transpose_first(1);

    TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor(add_desc);
    auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
    auto add_0 = op::Add("add_0");
    add_0.set_input_x1(layer_norm, "y");
    add_0.set_input_x2(add_const_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{24, 16, 4, 32, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionLnQKV") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(1);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}

TEST_F(attention_ln_qkv_fusion_test, attention_ln_qkv_fusion_test_5) {
    // layer_norm -> trans_data -> reformat pattern check failed
    ge::Graph graph("attention_ln_qkv_fusion_test5");

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 8;
    opti_compilation_info.soc_version = "Ascend710";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

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
    TensorDesc input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc input_desc_gamma(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc input_desc_beta(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    TensorDesc output_desc_mean(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_variance(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    layer_norm.update_input_desc_x(input_desc_x);
    layer_norm.update_input_desc_gamma(input_desc_gamma);
    layer_norm.update_input_desc_beta(input_desc_beta);
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.update_output_desc_mean(output_desc_mean);
    layer_norm.update_output_desc_variance(output_desc_variance);
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    auto end_op_mean = op::Cast("end_op_mean");
    end_op_mean.set_input_x(layer_norm, "mean");
    end_op_mean.set_attr_dst_type(0);

    auto end_op_variance = op::Cast("end_op_variance");
    end_op_variance.set_input_x(layer_norm, "variance");
    end_op_variance.set_attr_dst_type(1);

    auto trans_data_0 = op::TransData("trans_data_0");
    trans_data_0.set_input_src(layer_norm, "y");
    trans_data_0.set_attr_src_format("FRACTAL_NZ");
    trans_data_0.set_attr_dst_format("ND");

    auto trans_data_1 = op::TransData("trans_data_1");
    trans_data_1.set_input_src(trans_data_0);
    trans_data_1.set_attr_src_format("ND");
    trans_data_1.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_2 = op::TransData("trans_data_2");
    trans_data_2.set_input_src(trans_data_0);
    trans_data_2.set_attr_src_format("ND");
    trans_data_2.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_3 = op::TransData("trans_data_3");
    trans_data_3.set_input_src(trans_data_0);
    trans_data_3.set_attr_src_format("ND");
    trans_data_3.set_attr_dst_format("FRACTAL_NZ");

    TensorDesc add_desc_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor_1(add_desc_1);
    auto add_const_op_1 = op::Const("add_const_op_1").set_attr_value(add_tensor_1);
    auto add_1 = op::Add("add_1");
    add_1.set_input_x1(trans_data_0);
    add_1.set_input_x2(add_const_op_1);

    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    TensorDesc bias_query_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_query_tensor(bias_query_desc);
    auto bias_query = op::Const("bias_query").set_attr_value(bias_query_tensor);
    TensorDesc bias_key_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_key_tensor(bias_key_desc);
    auto bias_key = op::Const("bias_key").set_attr_value(bias_key_tensor);
    TensorDesc bias_value_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_value_tensor(bias_value_desc);
    auto bias_value = op::Const("bias_value").set_attr_value(bias_value_tensor);

    auto matmul_query = op::MatMulV2("matmul_query");
    matmul_query.set_input_x1(trans_data_1);
    matmul_query.set_input_x2(kernel_query);
    matmul_query.set_input_bias(bias_query);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(false);

    auto matmul_key = op::MatMulV2("matmul_key");
    matmul_key.set_input_x1(trans_data_2);
    matmul_key.set_input_x2(kernel_key);
    matmul_key.set_input_bias(bias_key);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(false);

    auto matmul_value = op::MatMulV2("matmul_value");
    matmul_value.set_input_x1(trans_data_3);
    matmul_value.set_input_x2(kernel_value);
    matmul_value.set_input_bias(bias_value);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(false);

    auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
    conf_transpose_query.set_input_x(matmul_query);
    conf_transpose_query.set_attr_perm({0, 2, 1, 3});
    conf_transpose_query.set_attr_shape({24, 512, 16, 64});
    conf_transpose_query.set_attr_transpose_first(1);

    auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
    conf_transpose_key.set_input_x(matmul_key);
    conf_transpose_key.set_attr_perm({0, 2, 1, 3});
    conf_transpose_key.set_attr_shape({24, 512, 16, 64});
    conf_transpose_key.set_attr_transpose_first(1);

    auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
    conf_transpose_value.set_input_x(matmul_value);
    conf_transpose_value.set_attr_perm({0, 2, 1, 3});
    conf_transpose_value.set_attr_shape({24, 512, 16, 64});
    conf_transpose_value.set_attr_transpose_first(1);

    TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor(add_desc);
    auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
    auto add_0 = op::Add("add_0");
    add_0.set_input_x1(layer_norm, "y");
    add_0.set_input_x2(add_const_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{24, 16, 4, 32, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionLnQKV") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(1);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}

TEST_F(attention_ln_qkv_fusion_test, attention_ln_qkv_fusion_test_6) {
    // nums of nodes connect with layer_norm output y no more than 3
    ge::Graph graph("attention_ln_qkv_fusion_test6");

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 8;
    opti_compilation_info.soc_version = "Ascend710";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

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
    TensorDesc input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc input_desc_gamma(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc input_desc_beta(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    TensorDesc output_desc_mean(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_variance(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    layer_norm.update_input_desc_x(input_desc_x);
    layer_norm.update_input_desc_gamma(input_desc_gamma);
    layer_norm.update_input_desc_beta(input_desc_beta);
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.update_output_desc_mean(output_desc_mean);
    layer_norm.update_output_desc_variance(output_desc_variance);
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    auto end_op_mean = op::Cast("end_op_mean");
    end_op_mean.set_input_x(layer_norm, "mean");
    end_op_mean.set_attr_dst_type(0);

    auto end_op_variance = op::Cast("end_op_variance");
    end_op_variance.set_input_x(layer_norm, "variance");
    end_op_variance.set_attr_dst_type(1);

    auto trans_data_0 = op::TransData("trans_data_0");
    trans_data_0.set_input_src(layer_norm, "y");
    trans_data_0.set_attr_src_format("FRACTAL_NZ");
    trans_data_0.set_attr_dst_format("ND");

    auto reformat_0 = op::ReFormat("reformat");
    reformat_0.set_input_x(trans_data_0);

    auto trans_data_1 = op::TransData("trans_data_1");
    trans_data_1.set_input_src(reformat_0);
    trans_data_1.set_attr_src_format("NHWC");
    trans_data_1.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_2 = op::TransData("trans_data_2");
    trans_data_2.set_input_src(reformat_0);
    trans_data_2.set_attr_src_format("NHWC");
    trans_data_2.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_3 = op::TransData("trans_data_3");
    trans_data_3.set_input_src(reformat_0);
    trans_data_3.set_attr_src_format("NHWC");
    trans_data_3.set_attr_dst_format("FRACTAL_NZ");

    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    TensorDesc bias_query_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_query_tensor(bias_query_desc);
    auto bias_query = op::Const("bias_query").set_attr_value(bias_query_tensor);
    TensorDesc bias_key_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_key_tensor(bias_key_desc);
    auto bias_key = op::Const("bias_key").set_attr_value(bias_key_tensor);
    TensorDesc bias_value_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_value_tensor(bias_value_desc);
    auto bias_value = op::Const("bias_value").set_attr_value(bias_value_tensor);

    auto matmul_query = op::MatMulV2("matmul_query");
    matmul_query.set_input_x1(trans_data_1);
    matmul_query.set_input_x2(kernel_query);
    matmul_query.set_input_bias(bias_query);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(false);

    auto matmul_key = op::MatMulV2("matmul_key");
    matmul_key.set_input_x1(trans_data_2);
    matmul_key.set_input_x2(kernel_key);
    matmul_key.set_input_bias(bias_key);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(false);

    auto matmul_value = op::MatMulV2("matmul_value");
    matmul_value.set_input_x1(trans_data_3);
    matmul_value.set_input_x2(kernel_value);
    matmul_value.set_input_bias(bias_value);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(false);

    auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
    conf_transpose_query.set_input_x(matmul_query);
    conf_transpose_query.set_attr_perm({0, 2, 1, 3});
    conf_transpose_query.set_attr_shape({24, 512, 16, 64});
    conf_transpose_query.set_attr_transpose_first(1);

    auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
    conf_transpose_key.set_input_x(matmul_key);
    conf_transpose_key.set_attr_perm({0, 2, 1, 3});
    conf_transpose_key.set_attr_shape({24, 512, 16, 64});
    conf_transpose_key.set_attr_transpose_first(1);

    auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
    conf_transpose_value.set_input_x(matmul_value);
    conf_transpose_value.set_attr_perm({0, 2, 1, 3});
    conf_transpose_value.set_attr_shape({24, 512, 16, 64});
    conf_transpose_value.set_attr_transpose_first(1);

    TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor(add_desc);
    auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
    auto add_0 = op::Add("add_0");
    add_0.set_input_x1(layer_norm, "y");
    add_0.set_input_x2(add_const_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{24, 16, 4, 32, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionLnQKV") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(1);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}

TEST_F(attention_ln_qkv_fusion_test, attention_ln_qkv_fusion_test_7) {
    // matmul_qkv's trans_a/trans_b check failed
    ge::Graph graph("attention_ln_qkv_fusion_test7");

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 8;
    opti_compilation_info.soc_version = "Ascend710";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

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
    TensorDesc input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc input_desc_gamma(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc input_desc_beta(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1020}));
    TensorDesc output_desc_mean(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_variance(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    layer_norm.update_input_desc_x(input_desc_x);
    layer_norm.update_input_desc_gamma(input_desc_gamma);
    layer_norm.update_input_desc_beta(input_desc_beta);
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.update_output_desc_mean(output_desc_mean);
    layer_norm.update_output_desc_variance(output_desc_variance);
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    auto end_op_mean = op::Cast("end_op_mean");
    end_op_mean.set_input_x(layer_norm, "mean");
    end_op_mean.set_attr_dst_type(0);

    auto end_op_variance = op::Cast("end_op_variance");
    end_op_variance.set_input_x(layer_norm, "variance");
    end_op_variance.set_attr_dst_type(1);

    auto trans_data_0 = op::TransData("trans_data_0");
    trans_data_0.set_input_src(layer_norm, "y");
    trans_data_0.set_attr_src_format("FRACTAL_NZ");
    trans_data_0.set_attr_dst_format("ND");

    auto reformat_0 = op::ReFormat("reformat");
    reformat_0.set_input_x(trans_data_0);

    auto trans_data_1 = op::TransData("trans_data_1");
    trans_data_1.set_input_src(reformat_0);
    trans_data_1.set_attr_src_format("NHWC");
    trans_data_1.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_2 = op::TransData("trans_data_2");
    trans_data_2.set_input_src(reformat_0);
    trans_data_2.set_attr_src_format("NHWC");
    trans_data_2.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_3 = op::TransData("trans_data_3");
    trans_data_3.set_input_src(reformat_0);
    trans_data_3.set_attr_src_format("NHWC");
    trans_data_3.set_attr_dst_format("FRACTAL_NZ");

    TensorDesc add_desc_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor_1(add_desc_1);
    auto add_const_op_1 = op::Const("add_const_op_1").set_attr_value(add_tensor_1);
    auto add_1 = op::Add("add_1");
    add_1.set_input_x1(reformat_0);
    add_1.set_input_x2(add_const_op_1);

    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    TensorDesc bias_query_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_query_tensor(bias_query_desc);
    auto bias_query = op::Const("bias_query").set_attr_value(bias_query_tensor);
    TensorDesc bias_key_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_key_tensor(bias_key_desc);
    auto bias_key = op::Const("bias_key").set_attr_value(bias_key_tensor);
    TensorDesc bias_value_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_value_tensor(bias_value_desc);
    auto bias_value = op::Const("bias_value").set_attr_value(bias_value_tensor);

    auto matmul_query = op::MatMulV2("matmul_query");
    matmul_query.set_input_x1(trans_data_1);
    matmul_query.set_input_x2(kernel_query);
    matmul_query.set_input_bias(bias_query);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(true);

    auto matmul_key = op::MatMulV2("matmul_key");
    matmul_key.set_input_x1(trans_data_2);
    matmul_key.set_input_x2(kernel_key);
    matmul_key.set_input_bias(bias_key);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(true);

    auto matmul_value = op::MatMulV2("matmul_value");
    matmul_value.set_input_x1(trans_data_3);
    matmul_value.set_input_x2(kernel_value);
    matmul_value.set_input_bias(bias_value);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(true);

    auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
    conf_transpose_query.set_input_x(matmul_query);
    conf_transpose_query.set_attr_perm({0, 2, 1, 3});
    conf_transpose_query.set_attr_shape({24, 512, 16, 64});
    conf_transpose_query.set_attr_transpose_first(1);

    auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
    conf_transpose_key.set_input_x(matmul_key);
    conf_transpose_key.set_attr_perm({0, 2, 1, 3});
    conf_transpose_key.set_attr_shape({24, 512, 16, 64});
    conf_transpose_key.set_attr_transpose_first(1);

    auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
    conf_transpose_value.set_input_x(matmul_value);
    conf_transpose_value.set_attr_perm({0, 2, 1, 3});
    conf_transpose_value.set_attr_shape({24, 512, 16, 64});
    conf_transpose_value.set_attr_transpose_first(1);

    TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor(add_desc);
    auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
    auto add_0 = op::Add("add_0");
    add_0.set_input_x1(layer_norm, "y");
    add_0.set_input_x2(add_const_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{24, 16, 4, 32, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionLnQKV") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(1);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}

TEST_F(attention_ln_qkv_fusion_test, attention_ln_qkv_fusion_test_8) {
    ge::Graph graph("attention_ln_qkv_fusion_test8");

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 8;
    opti_compilation_info.soc_version = "Ascend710";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

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
    TensorDesc input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc input_desc_gamma(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc input_desc_beta(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    TensorDesc output_desc_mean(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_variance(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    layer_norm.update_input_desc_x(input_desc_x);
    layer_norm.update_input_desc_gamma(input_desc_gamma);
    layer_norm.update_input_desc_beta(input_desc_beta);
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.update_output_desc_mean(output_desc_mean);
    layer_norm.update_output_desc_variance(output_desc_variance);
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    auto end_op_mean = op::Cast("end_op_mean");
    end_op_mean.set_input_x(layer_norm, "mean");
    end_op_mean.set_attr_dst_type(0);

    auto end_op_variance = op::Cast("end_op_variance");
    end_op_variance.set_input_x(layer_norm, "variance");
    end_op_variance.set_attr_dst_type(1);

    auto trans_data_0 = op::TransData("trans_data_0");
    TensorDesc trans_data_in_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc trans_data_out_desc(ge::Shape({12288, 1024}), FORMAT_ND, DT_FLOAT16);
    trans_data_0.update_input_desc_src(trans_data_in_desc);
    trans_data_0.update_output_desc_dst(trans_data_out_desc);
    trans_data_0.set_input_src(layer_norm, "y");
    trans_data_0.set_attr_src_format("FRACTAL_NZ");
    trans_data_0.set_attr_dst_format("ND");

    auto reformat_0 = op::ReFormat("reformat");
    TensorDesc reformat_in_desc(ge::Shape({12288, 1024}), FORMAT_ND, DT_FLOAT16);
    TensorDesc reformat_out_desc(ge::Shape({12288, 1024}), FORMAT_NHWC, DT_FLOAT16);
    reformat_0.update_input_desc_x(reformat_in_desc);
    reformat_0.update_output_desc_y(reformat_out_desc);
    reformat_0.set_input_x(trans_data_0);

    auto trans_data_1 = op::TransData("trans_data_1");
    TensorDesc trans_data_in_desc1(ge::Shape({12288, 1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc trans_data_out_desc1(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    trans_data_1.update_input_desc_src(trans_data_in_desc1);
    trans_data_1.update_output_desc_dst(trans_data_out_desc1);
    trans_data_1.set_input_src(reformat_0);
    trans_data_1.set_attr_src_format("NHWC");
    trans_data_1.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_2 = op::TransData("trans_data_2");
    TensorDesc trans_data_in_desc2(ge::Shape({12288, 1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc trans_data_out_desc2(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    trans_data_2.update_input_desc_src(trans_data_in_desc2);
    trans_data_2.update_output_desc_dst(trans_data_out_desc2);
    trans_data_2.set_input_src(reformat_0);
    trans_data_2.set_attr_src_format("NHWC");
    trans_data_2.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_3 = op::TransData("trans_data_3");
    TensorDesc trans_data_in_desc3(ge::Shape({12288, 1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc trans_data_out_desc3(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    trans_data_3.update_input_desc_src(trans_data_in_desc3);
    trans_data_3.update_output_desc_dst(trans_data_out_desc3);
    trans_data_3.set_input_src(reformat_0);
    trans_data_3.set_attr_src_format("NHWC");
    trans_data_3.set_attr_dst_format("FRACTAL_NZ");

    TensorDesc add_desc_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor_1(add_desc_1);
    auto add_const_op_1 = op::Const("add_const_op_1").set_attr_value(add_tensor_1);
    auto add_1 = op::Add("add_1");
    TensorDesc add_input_desc_x1_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc add_input_desc_x2_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc add_output_desc_y_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    add_1.update_input_desc_x1(add_input_desc_x1_1);
    add_1.update_input_desc_x2(add_input_desc_x2_1);
    add_1.update_output_desc_y(add_output_desc_y_1);
    add_1.set_input_x1(reformat_0);
    add_1.set_input_x2(add_const_op_1);

    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    TensorDesc bias_query_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_query_tensor(bias_query_desc);
    auto bias_query = op::Const("bias_query").set_attr_value(bias_query_tensor);
    TensorDesc bias_key_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_key_tensor(bias_key_desc);
    auto bias_key = op::Const("bias_key").set_attr_value(bias_key_tensor);
    TensorDesc bias_value_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_value_tensor(bias_value_desc);
    auto bias_value = op::Const("bias_value").set_attr_value(bias_value_tensor);

    auto matmul_query = op::MatMulV2("matmul_query");
    TensorDesc mm_q_input_desc_x1(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_q_input_desc_x2(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_q_input_desc_bias(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    TensorDesc mm_q_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_query.update_input_desc_x1(mm_q_input_desc_x1);
    matmul_query.update_input_desc_x2(mm_q_input_desc_x2);
    matmul_query.update_input_desc_bias(mm_q_input_desc_bias);
    matmul_query.update_output_desc_y(mm_q_output_desc_y);
    matmul_query.set_input_x1(trans_data_1);
    matmul_query.set_input_x2(kernel_query);
    matmul_query.set_input_bias(bias_query);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(false);

    auto matmul_key = op::MatMulV2("matmul_key");
    TensorDesc mm_k_input_desc_x1(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_k_input_desc_x2(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_k_input_desc_bias(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    TensorDesc mm_k_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_key.update_input_desc_x1(mm_k_input_desc_x1);
    matmul_key.update_input_desc_x2(mm_k_input_desc_x2);
    matmul_key.update_input_desc_bias(mm_k_input_desc_bias);
    matmul_key.update_output_desc_y(mm_k_output_desc_y);
    matmul_key.set_input_x1(trans_data_2);
    matmul_key.set_input_x2(kernel_key);
    matmul_key.set_input_bias(bias_key);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(false);

    auto matmul_value = op::MatMulV2("matmul_value");
    TensorDesc mm_v_input_desc_x1(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_v_input_desc_x2(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_v_input_desc_bias(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    TensorDesc mm_v_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_value.update_input_desc_x1(mm_v_input_desc_x1);
    matmul_value.update_input_desc_x2(mm_v_input_desc_x2);
    matmul_value.update_input_desc_bias(mm_v_input_desc_bias);
    matmul_value.update_output_desc_y(mm_v_output_desc_y);
    matmul_value.set_input_x1(trans_data_3);
    matmul_value.set_input_x2(kernel_value);
    matmul_value.set_input_bias(bias_value);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(false);

    auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
    TensorDesc conf_trans_query_input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc conf_trans_query_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    conf_transpose_query.update_input_desc_x(conf_trans_query_input_desc_x);
    conf_transpose_query.update_output_desc_y(conf_trans_query_output_desc_y);
    conf_transpose_query.set_input_x(matmul_query);
    conf_transpose_query.set_attr_perm({0, 2, 1, 3});
    conf_transpose_query.set_attr_shape({24, 512, 16, 64});
    conf_transpose_query.set_attr_transpose_first(1);

    auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
    TensorDesc conf_trans_key_input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc conf_trans_key_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    conf_transpose_key.update_input_desc_x(conf_trans_key_input_desc_x);
    conf_transpose_key.update_output_desc_y(conf_trans_key_output_desc_y);
    conf_transpose_key.set_input_x(matmul_key);
    conf_transpose_key.set_attr_perm({0, 2, 1, 3});
    conf_transpose_key.set_attr_shape({24, 512, 16, 64});
    conf_transpose_key.set_attr_transpose_first(1);

    auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
    TensorDesc conf_trans_value_input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc conf_trans_value_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    conf_transpose_value.update_input_desc_x(conf_trans_value_input_desc_x);
    conf_transpose_value.update_output_desc_y(conf_trans_value_output_desc_y);
    conf_transpose_value.set_input_x(matmul_value);
    conf_transpose_value.set_attr_perm({0, 2, 1, 3});
    conf_transpose_value.set_attr_shape({24, 512, 16, 64});
    conf_transpose_value.set_attr_transpose_first(1);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{24, 16, 4, 32, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionLnQKV") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(1);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}

TEST_F(attention_ln_qkv_fusion_test, attention_ln_qkv_fusion_test_9) {
    ge::Graph graph("attention_ln_qkv_fusion_test9");

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 8;
    opti_compilation_info.soc_version = "Ascend710";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    auto input_x_data = op::Data("input_x_data");
    TensorDesc input_x_desc(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_x_data.update_input_desc_x(input_x_desc);
    input_x_data.update_output_desc_y(input_x_desc);

    TensorDesc gamma_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor gamma_tensor(gamma_desc);
    auto gamma = op::Const("gamma").set_attr_value(gamma_tensor);

    TensorDesc beta_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    Tensor beta_tensor(beta_desc);
    auto beta = op::Const("beta").set_attr_value(beta_tensor);

    auto layer_norm = op::LayerNorm("layer_norm_0");
    TensorDesc input_desc_x(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc input_desc_gamma(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc input_desc_beta(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_y(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({2048, 1024}));
    TensorDesc output_desc_mean(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_variance(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    layer_norm.update_input_desc_x(input_desc_x);
    layer_norm.update_input_desc_gamma(input_desc_gamma);
    layer_norm.update_input_desc_beta(input_desc_beta);
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.update_output_desc_mean(output_desc_mean);
    layer_norm.update_output_desc_variance(output_desc_variance);
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    auto end_op_mean = op::Cast("end_op_mean");
    end_op_mean.set_input_x(layer_norm, "mean");
    end_op_mean.set_attr_dst_type(0);

    auto end_op_variance = op::Cast("end_op_variance");
    end_op_variance.set_input_x(layer_norm, "variance");
    end_op_variance.set_attr_dst_type(1);

    auto trans_data_0 = op::TransData("trans_data_0");
    TensorDesc trans_data_in_desc(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc trans_data_out_desc(ge::Shape({2048, 1024}), FORMAT_ND, DT_FLOAT16);
    trans_data_0.update_input_desc_src(trans_data_in_desc);
    trans_data_0.update_output_desc_dst(trans_data_out_desc);
    trans_data_0.set_input_src(layer_norm, "y");
    trans_data_0.set_attr_src_format("FRACTAL_NZ");
    trans_data_0.set_attr_dst_format("ND");

    auto reformat_0 = op::ReFormat("reformat");
    TensorDesc reformat_in_desc(ge::Shape({2048, 1024}), FORMAT_ND, DT_FLOAT16);
    TensorDesc reformat_out_desc(ge::Shape({2048, 1024}), FORMAT_NHWC, DT_FLOAT16);
    reformat_0.update_input_desc_x(reformat_in_desc);
    reformat_0.update_output_desc_y(reformat_out_desc);
    reformat_0.set_input_x(trans_data_0);

    auto trans_data_1 = op::TransData("trans_data_1");
    TensorDesc trans_data_in_desc1(ge::Shape({2048, 1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc trans_data_out_desc1(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    trans_data_1.update_input_desc_src(trans_data_in_desc1);
    trans_data_1.update_output_desc_dst(trans_data_out_desc1);
    trans_data_1.set_input_src(reformat_0);
    trans_data_1.set_attr_src_format("NHWC");
    trans_data_1.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_2 = op::TransData("trans_data_2");
    TensorDesc trans_data_in_desc2(ge::Shape({2048, 1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc trans_data_out_desc2(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    trans_data_2.update_input_desc_src(trans_data_in_desc2);
    trans_data_2.update_output_desc_dst(trans_data_out_desc2);
    trans_data_2.set_input_src(reformat_0);
    trans_data_2.set_attr_src_format("NHWC");
    trans_data_2.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_3 = op::TransData("trans_data_3");
    TensorDesc trans_data_in_desc3(ge::Shape({2048, 1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc trans_data_out_desc3(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    trans_data_3.update_input_desc_src(trans_data_in_desc3);
    trans_data_3.update_output_desc_dst(trans_data_out_desc3);
    trans_data_3.set_input_src(reformat_0);
    trans_data_3.set_attr_src_format("NHWC");
    trans_data_3.set_attr_dst_format("FRACTAL_NZ");

    TensorDesc add_desc_1(ge::Shape({2048, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor_1(add_desc_1);
    auto add_const_op_1 = op::Const("add_const_op_1").set_attr_value(add_tensor_1);
    auto add_1 = op::Add("add_1");
    TensorDesc add_input_desc_x1_1(ge::Shape({2048, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc add_input_desc_x2_1(ge::Shape({2048, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc add_output_desc_y_1(ge::Shape({2048, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    add_1.update_input_desc_x1(add_input_desc_x1_1);
    add_1.update_input_desc_x2(add_input_desc_x2_1);
    add_1.update_output_desc_y(add_output_desc_y_1);
    add_1.set_input_x1(reformat_0);
    add_1.set_input_x2(add_const_op_1);

    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    TensorDesc bias_query_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_query_tensor(bias_query_desc);
    auto bias_query = op::Const("bias_query").set_attr_value(bias_query_tensor);
    TensorDesc bias_key_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_key_tensor(bias_key_desc);
    auto bias_key = op::Const("bias_key").set_attr_value(bias_key_tensor);
    TensorDesc bias_value_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_value_tensor(bias_value_desc);
    auto bias_value = op::Const("bias_value").set_attr_value(bias_value_tensor);

    auto matmul_query = op::MatMulV2("matmul_query");
    TensorDesc mm_q_input_desc_x1(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_q_input_desc_x2(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_q_input_desc_bias(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    TensorDesc mm_q_output_desc_y(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_query.update_input_desc_x1(mm_q_input_desc_x1);
    matmul_query.update_input_desc_x2(mm_q_input_desc_x2);
    matmul_query.update_input_desc_bias(mm_q_input_desc_bias);
    matmul_query.update_output_desc_y(mm_q_output_desc_y);
    matmul_query.set_input_x1(trans_data_1);
    matmul_query.set_input_x2(kernel_query);
    matmul_query.set_input_bias(bias_query);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(false);

    auto matmul_key = op::MatMulV2("matmul_key");
    TensorDesc mm_k_input_desc_x1(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_k_input_desc_x2(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_k_input_desc_bias(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    TensorDesc mm_k_output_desc_y(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_key.update_input_desc_x1(mm_k_input_desc_x1);
    matmul_key.update_input_desc_x2(mm_k_input_desc_x2);
    matmul_key.update_input_desc_bias(mm_k_input_desc_bias);
    matmul_key.update_output_desc_y(mm_k_output_desc_y);
    matmul_key.set_input_x1(trans_data_2);
    matmul_key.set_input_x2(kernel_key);
    matmul_key.set_input_bias(bias_key);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(false);

    auto matmul_value = op::MatMulV2("matmul_value");
    TensorDesc mm_v_input_desc_x1(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_v_input_desc_x2(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_v_input_desc_bias(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    TensorDesc mm_v_output_desc_y(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_value.update_input_desc_x1(mm_v_input_desc_x1);
    matmul_value.update_input_desc_x2(mm_v_input_desc_x2);
    matmul_value.update_input_desc_bias(mm_v_input_desc_bias);
    matmul_value.update_output_desc_y(mm_v_output_desc_y);
    matmul_value.set_input_x1(trans_data_3);
    matmul_value.set_input_x2(kernel_value);
    matmul_value.set_input_bias(bias_value);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(false);

    auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
    TensorDesc conf_trans_query_input_desc_x(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc conf_trans_query_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    conf_transpose_query.update_input_desc_x(conf_trans_query_input_desc_x);
    conf_transpose_query.update_output_desc_y(conf_trans_query_output_desc_y);
    conf_transpose_query.set_input_x(matmul_query);
    conf_transpose_query.set_attr_perm({0, 2, 1, 3});
    conf_transpose_query.set_attr_shape({24, 512, 16, 64});
    conf_transpose_query.set_attr_transpose_first(1);

    auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
    TensorDesc conf_trans_key_input_desc_x(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc conf_trans_key_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    conf_transpose_key.update_input_desc_x(conf_trans_key_input_desc_x);
    conf_transpose_key.update_output_desc_y(conf_trans_key_output_desc_y);
    conf_transpose_key.set_input_x(matmul_key);
    conf_transpose_key.set_attr_perm({0, 2, 1, 3});
    conf_transpose_key.set_attr_shape({24, 512, 16, 64});
    conf_transpose_key.set_attr_transpose_first(1);

    auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
    TensorDesc conf_trans_value_input_desc_x(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc conf_trans_value_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    conf_transpose_value.update_input_desc_x(conf_trans_value_input_desc_x);
    conf_transpose_value.update_output_desc_y(conf_trans_value_output_desc_y);
    conf_transpose_value.set_input_x(matmul_value);
    conf_transpose_value.set_attr_perm({0, 2, 1, 3});
    conf_transpose_value.set_attr_shape({24, 512, 16, 64});
    conf_transpose_value.set_attr_transpose_first(1);

    TensorDesc add_desc(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor(add_desc);
    auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
    auto add_0 = op::Add("add_0");
    TensorDesc add_input_desc_x1(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc add_input_desc_x2(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc add_output_desc_y(ge::Shape({64, 128, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    add_0.update_input_desc_x1(add_input_desc_x1);
    add_0.update_input_desc_x2(add_input_desc_x2);
    add_0.update_output_desc_y(add_output_desc_y);
    add_0.set_input_x1(layer_norm, "y");
    add_0.set_input_x2(add_const_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{24, 16, 4, 32, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionLnQKV") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(1);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}

TEST_F(attention_ln_qkv_fusion_test, attention_ln_qkv_fusion_test_10) {
    ge::Graph graph("attention_ln_qkv_fusion_test10");

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 2;
    opti_compilation_info.soc_version = "Ascend310";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend310"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

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
    TensorDesc input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc input_desc_gamma(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc input_desc_beta(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
    TensorDesc output_desc_mean(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc output_desc_variance(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
    layer_norm.update_input_desc_x(input_desc_x);
    layer_norm.update_input_desc_gamma(input_desc_gamma);
    layer_norm.update_input_desc_beta(input_desc_beta);
    layer_norm.update_output_desc_y(output_desc_y);
    layer_norm.update_output_desc_mean(output_desc_mean);
    layer_norm.update_output_desc_variance(output_desc_variance);
    layer_norm.set_input_x(input_x_data);
    layer_norm.set_input_gamma(gamma);
    layer_norm.set_input_beta(beta);
    layer_norm.set_attr_begin_norm_axis(1);
    layer_norm.set_attr_begin_params_axis(-1);

    auto end_op_mean = op::Cast("end_op_mean");
    end_op_mean.set_input_x(layer_norm, "mean");
    end_op_mean.set_attr_dst_type(0);

    auto end_op_variance = op::Cast("end_op_variance");
    end_op_variance.set_input_x(layer_norm, "variance");
    end_op_variance.set_attr_dst_type(1);

    auto trans_data_0 = op::TransData("trans_data_0");
    TensorDesc trans_data_in_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc trans_data_out_desc(ge::Shape({12288, 1024}), FORMAT_ND, DT_FLOAT16);
    trans_data_0.update_input_desc_src(trans_data_in_desc);
    trans_data_0.update_output_desc_dst(trans_data_out_desc);
    trans_data_0.set_input_src(layer_norm, "y");
    trans_data_0.set_attr_src_format("FRACTAL_NZ");
    trans_data_0.set_attr_dst_format("ND");

    auto reformat_0 = op::ReFormat("reformat");
    TensorDesc reformat_in_desc(ge::Shape({12288, 1024}), FORMAT_ND, DT_FLOAT16);
    TensorDesc reformat_out_desc(ge::Shape({12288, 1024}), FORMAT_NHWC, DT_FLOAT16);
    reformat_0.update_input_desc_x(reformat_in_desc);
    reformat_0.update_output_desc_y(reformat_out_desc);
    reformat_0.set_input_x(trans_data_0);

    auto trans_data_1 = op::TransData("trans_data_1");
    TensorDesc trans_data_in_desc1(ge::Shape({12288, 1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc trans_data_out_desc1(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    trans_data_1.update_input_desc_src(trans_data_in_desc1);
    trans_data_1.update_output_desc_dst(trans_data_out_desc1);
    trans_data_1.set_input_src(reformat_0);
    trans_data_1.set_attr_src_format("NHWC");
    trans_data_1.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_2 = op::TransData("trans_data_2");
    TensorDesc trans_data_in_desc2(ge::Shape({12288, 1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc trans_data_out_desc2(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    trans_data_2.update_input_desc_src(trans_data_in_desc2);
    trans_data_2.update_output_desc_dst(trans_data_out_desc2);
    trans_data_2.set_input_src(reformat_0);
    trans_data_2.set_attr_src_format("NHWC");
    trans_data_2.set_attr_dst_format("FRACTAL_NZ");

    auto trans_data_3 = op::TransData("trans_data_3");
    TensorDesc trans_data_in_desc3(ge::Shape({12288, 1024}), FORMAT_NHWC, DT_FLOAT16);
    TensorDesc trans_data_out_desc3(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    trans_data_3.update_input_desc_src(trans_data_in_desc3);
    trans_data_3.update_output_desc_dst(trans_data_out_desc3);
    trans_data_3.set_input_src(reformat_0);
    trans_data_3.set_attr_src_format("NHWC");
    trans_data_3.set_attr_dst_format("FRACTAL_NZ");

    TensorDesc add_desc_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor_1(add_desc_1);
    auto add_const_op_1 = op::Const("add_const_op_1").set_attr_value(add_tensor_1);
    auto add_1 = op::Add("add_1");
    TensorDesc add_input_desc_x1_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc add_input_desc_x2_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc add_output_desc_y_1(ge::Shape({12288, 1024}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    add_1.update_input_desc_x1(add_input_desc_x1_1);
    add_1.update_input_desc_x2(add_input_desc_x2_1);
    add_1.update_output_desc_y(add_output_desc_y_1);
    add_1.set_input_x1(reformat_0);
    add_1.set_input_x2(add_const_op_1);

    TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_query_tensor(kernel_query_desc);
    auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
    TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_key_tensor(kernel_key_desc);
    auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
    TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor kernel_value_tensor(kernel_value_desc);
    auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

    TensorDesc bias_query_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_query_tensor(bias_query_desc);
    auto bias_query = op::Const("bias_query").set_attr_value(bias_query_tensor);
    TensorDesc bias_key_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_key_tensor(bias_key_desc);
    auto bias_key = op::Const("bias_key").set_attr_value(bias_key_tensor);
    TensorDesc bias_value_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    Tensor bias_value_tensor(bias_value_desc);
    auto bias_value = op::Const("bias_value").set_attr_value(bias_value_tensor);

    auto matmul_query = op::MatMulV2("matmul_query");
    TensorDesc mm_q_input_desc_x1(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_q_input_desc_x2(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_q_input_desc_bias(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    TensorDesc mm_q_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_query.update_input_desc_x1(mm_q_input_desc_x1);
    matmul_query.update_input_desc_x2(mm_q_input_desc_x2);
    matmul_query.update_input_desc_bias(mm_q_input_desc_bias);
    matmul_query.update_output_desc_y(mm_q_output_desc_y);
    matmul_query.set_input_x1(trans_data_1);
    matmul_query.set_input_x2(kernel_query);
    matmul_query.set_input_bias(bias_query);
    matmul_query.set_attr_transpose_x1(false);
    matmul_query.set_attr_transpose_x2(false);

    auto matmul_key = op::MatMulV2("matmul_key");
    TensorDesc mm_k_input_desc_x1(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_k_input_desc_x2(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_k_input_desc_bias(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    TensorDesc mm_k_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_key.update_input_desc_x1(mm_k_input_desc_x1);
    matmul_key.update_input_desc_x2(mm_k_input_desc_x2);
    matmul_key.update_input_desc_bias(mm_k_input_desc_bias);
    matmul_key.update_output_desc_y(mm_k_output_desc_y);
    matmul_key.set_input_x1(trans_data_2);
    matmul_key.set_input_x2(kernel_key);
    matmul_key.set_input_bias(bias_key);
    matmul_key.set_attr_transpose_x1(false);
    matmul_key.set_attr_transpose_x2(false);

    auto matmul_value = op::MatMulV2("matmul_value");
    TensorDesc mm_v_input_desc_x1(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_v_input_desc_x2(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc mm_v_input_desc_bias(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
    TensorDesc mm_v_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    matmul_value.update_input_desc_x1(mm_v_input_desc_x1);
    matmul_value.update_input_desc_x2(mm_v_input_desc_x2);
    matmul_value.update_input_desc_bias(mm_v_input_desc_bias);
    matmul_value.update_output_desc_y(mm_v_output_desc_y);
    matmul_value.set_input_x1(trans_data_3);
    matmul_value.set_input_x2(kernel_value);
    matmul_value.set_input_bias(bias_value);
    matmul_value.set_attr_transpose_x1(false);
    matmul_value.set_attr_transpose_x2(false);

    auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
    TensorDesc conf_trans_query_input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc conf_trans_query_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    conf_transpose_query.update_input_desc_x(conf_trans_query_input_desc_x);
    conf_transpose_query.update_output_desc_y(conf_trans_query_output_desc_y);
    conf_transpose_query.set_input_x(matmul_query);
    conf_transpose_query.set_attr_perm({0, 2, 1, 3});
    conf_transpose_query.set_attr_shape({24, 512, 16, 64});
    conf_transpose_query.set_attr_transpose_first(1);

    auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
    TensorDesc conf_trans_key_input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc conf_trans_key_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    conf_transpose_key.update_input_desc_x(conf_trans_key_input_desc_x);
    conf_transpose_key.update_output_desc_y(conf_trans_key_output_desc_y);
    conf_transpose_key.set_input_x(matmul_key);
    conf_transpose_key.set_attr_perm({0, 2, 1, 3});
    conf_transpose_key.set_attr_shape({24, 512, 16, 64});
    conf_transpose_key.set_attr_transpose_first(1);

    auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
    TensorDesc conf_trans_value_input_desc_x(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc conf_trans_value_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    conf_transpose_value.update_input_desc_x(conf_trans_value_input_desc_x);
    conf_transpose_value.update_output_desc_y(conf_trans_value_output_desc_y);
    conf_transpose_value.set_input_x(matmul_value);
    conf_transpose_value.set_attr_perm({0, 2, 1, 3});
    conf_transpose_value.set_attr_shape({24, 512, 16, 64});
    conf_transpose_value.set_attr_transpose_first(1);

    TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Tensor add_tensor(add_desc);
    auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
    auto add_0 = op::Add("add_0");
    TensorDesc add_input_desc_x1(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc add_input_desc_x2(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc add_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    add_0.update_input_desc_x1(add_input_desc_x1);
    add_0.update_input_desc_x2(add_input_desc_x2);
    add_0.update_output_desc_y(add_output_desc_y);
    add_0.set_input_x1(layer_norm, "y");
    add_0.set_input_x2(add_const_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{24, 16, 4, 32, 16, 16};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AttentionLnQKV") {
            findOp = true;
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(1);
            std::vector<int64_t> dims = outputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}