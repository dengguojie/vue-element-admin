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

class attention_ln_qkv_onnx_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
  std::cout << "attention_ln_qkv_onnx_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
  std::cout << "attention_ln_qkv_onnx_fusion_test TearDown" << std::endl;
  }
};

void SetLayernorm(op::LayerNorm& layer_norm, op::Data input_x_data,
                  op::Const& gamma, op::Const& beta,
                  TensorDesc& input_desc_x, TensorDesc& input_desc_gamma,
                  TensorDesc& input_desc_beta, TensorDesc& output_desc_y,
                  TensorDesc& output_desc_mean, TensorDesc& output_desc_variance) {
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
}

void SetMatmul(op::BatchMatMulV2& matmul_op, TensorDesc& output_desc_y,
               op::LayerNorm& layer_norm, op::Const kernel, op::Const bias) {
  matmul_op.update_output_desc_y(output_desc_y);
  matmul_op.set_input_x1_by_name(layer_norm, "y");
  matmul_op.set_input_x2(kernel);
  matmul_op.set_input_bias(bias);
}

void SetConfusionTransposeD(op::ConfusionTransposeD& conf_transpose_op,
                            TensorDesc& output_desc_y, op::BatchMatMulV2& matmul_op,
                            ge::Operator::OpListInt& perm, ge::Operator::OpListInt& shape) {
  conf_transpose_op.update_output_desc_y(output_desc_y);
  conf_transpose_op.set_input_x(matmul_op);
  conf_transpose_op.set_attr_perm(perm);
  conf_transpose_op.set_attr_shape(shape);
  conf_transpose_op.set_attr_transpose_first(1);
}


TEST_F(attention_ln_qkv_onnx_fusion_test, attention_ln_qkv_onnx_fusion_success_case) {
  // success testcase
  ge::Graph graph("attention_ln_qkv_onnx_fusion_test1");

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 8;
  opti_compilation_info.soc_version = "soc_version";
  platform_info.ai_core_intrinsic_dtype_map["Intrinsic_vln"] = {"float32"};
  fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
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
  SetLayernorm(layer_norm, input_x_data, gamma, beta,
               input_desc_x, input_desc_gamma, input_desc_beta,
               output_desc_y, output_desc_mean, output_desc_variance);

  auto end_op_mean = op::Cast("end_op_mean");
  end_op_mean.set_input_x_by_name(layer_norm, "mean");
  end_op_mean.set_attr_dst_type(0);

  auto end_op_variance = op::Cast("end_op_variance");
  end_op_variance.set_input_x_by_name(layer_norm, "variance");
  end_op_variance.set_attr_dst_type(1);

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

  auto matmul_query = op::BatchMatMulV2("matmul_query");
  TensorDesc mm_q_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  mm_q_output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
  SetMatmul(matmul_query, mm_q_output_desc_y, layer_norm, kernel_query, bias_query);

  auto matmul_key = op::BatchMatMulV2("matmul_key");
  TensorDesc mm_k_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_key, mm_k_output_desc_y, layer_norm, kernel_key, bias_key);

  auto matmul_value = op::BatchMatMulV2("matmul_value");
  TensorDesc mm_v_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_value, mm_v_output_desc_y, layer_norm, kernel_value, bias_value);

  auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
  TensorDesc conf_trans_query_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_query_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_query_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_query, conf_trans_query_output_desc_y,
                         matmul_query, conf_transpose_query_perm, conf_transpose_query_shape);

  auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
  TensorDesc conf_trans_key_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_key_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_key_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_key, conf_trans_key_output_desc_y,
                         matmul_key, conf_transpose_key_perm, conf_transpose_key_shape);

  auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
  TensorDesc conf_trans_value_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_value_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_value_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_value, conf_trans_value_output_desc_y,
                         matmul_value, conf_transpose_value_perm, conf_transpose_value_shape);

  TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor add_tensor(add_desc);
  auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
  auto add_0 = op::Add("add_0");
  TensorDesc add_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  add_0.update_output_desc_y(add_output_desc_y);
  add_0.set_input_x1_by_name(layer_norm, "y");
  add_0.set_input_x2(add_const_op);

  std::vector<Operator> inputs{input_x_data};
  std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVONNXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

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

TEST_F(attention_ln_qkv_onnx_fusion_test, attention_ln_qkv_onnx_fusion_layernorm_outputs_error) {
  // num of layernorm outputs not equal to 4
  ge::Graph graph("attention_ln_qkv_onnx_fusion_test2");

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 8;
  opti_compilation_info.soc_version = "soc_version";
  platform_info.ai_core_intrinsic_dtype_map["Intrinsic_vln"] = {"float32"};
  fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
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
  SetLayernorm(layer_norm, input_x_data, gamma, beta,
               input_desc_x, input_desc_gamma, input_desc_beta,
               output_desc_y, output_desc_mean, output_desc_variance);

  auto end_op_mean = op::Cast("end_op_mean");
  end_op_mean.set_input_x_by_name(layer_norm, "mean");
  end_op_mean.set_attr_dst_type(0);

  auto end_op_variance = op::Cast("end_op_variance");
  end_op_variance.set_input_x_by_name(layer_norm, "variance");
  end_op_variance.set_attr_dst_type(1);

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

  auto matmul_query = op::BatchMatMulV2("matmul_query");
  TensorDesc mm_q_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  mm_q_output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
  SetMatmul(matmul_query, mm_q_output_desc_y, layer_norm, kernel_query, bias_query);

  auto matmul_key = op::BatchMatMulV2("matmul_key");
  TensorDesc mm_k_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_key, mm_k_output_desc_y, layer_norm, kernel_key, bias_key);

  auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
  TensorDesc conf_trans_query_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_query_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_query_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_query, conf_trans_query_output_desc_y,
                         matmul_query, conf_transpose_query_perm, conf_transpose_query_shape);

  auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
  TensorDesc conf_trans_key_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_key_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_key_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_key, conf_trans_key_output_desc_y,
                         matmul_key, conf_transpose_key_perm, conf_transpose_key_shape);

  TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor add_tensor(add_desc);
  auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
  auto add_0 = op::Add("add_0");
  TensorDesc add_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  add_0.update_output_desc_y(add_output_desc_y);
  add_0.set_input_x1_by_name(layer_norm, "y");
  add_0.set_input_x2(add_const_op);

  std::vector<Operator> inputs{input_x_data};
  std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVONNXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

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

TEST_F(attention_ln_qkv_onnx_fusion_test, attention_ln_qkv_onnx_fusion_layernorm_bmm_pattern_error) {
  // layernorm -> batchmatmulv2 pattern check failed
  ge::Graph graph("attention_ln_qkv_onnx_fusion_test3");

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 8;
  opti_compilation_info.soc_version = "soc_version";
  platform_info.ai_core_intrinsic_dtype_map["Intrinsic_vln"] = {"float32"};
  fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
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
  SetLayernorm(layer_norm, input_x_data, gamma, beta,
               input_desc_x, input_desc_gamma, input_desc_beta,
               output_desc_y, output_desc_mean, output_desc_variance);

  auto end_op_mean = op::Cast("end_op_mean");
  end_op_mean.set_input_x_by_name(layer_norm, "mean");
  end_op_mean.set_attr_dst_type(0);

  auto end_op_variance = op::Cast("end_op_variance");
  end_op_variance.set_input_x_by_name(layer_norm, "variance");
  end_op_variance.set_attr_dst_type(1);

  TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor add_tensor(add_desc);
  auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
  auto add_0 = op::Add("add_0");
  TensorDesc add_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  add_0.update_output_desc_y(add_output_desc_y);
  add_0.set_input_x1_by_name(layer_norm, "y");
  add_0.set_input_x2(add_const_op);

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

  auto matmul_query = op::BatchMatMulV2("matmul_query");
  TensorDesc mm_q_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  mm_q_output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
  SetMatmul(matmul_query, mm_q_output_desc_y, layer_norm, kernel_query, bias_query);

  auto matmul_key = op::BatchMatMulV2("matmul_key");
  TensorDesc mm_k_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_key, mm_k_output_desc_y, layer_norm, kernel_key, bias_key);

  auto matmul_value = op::BatchMatMulV2("matmul_value");
  TensorDesc mm_v_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_value, mm_v_output_desc_y, layer_norm, kernel_value, bias_value);

  auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
  TensorDesc conf_trans_query_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_query_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_query_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_query, conf_trans_query_output_desc_y,
                         matmul_query, conf_transpose_query_perm, conf_transpose_query_shape);

  auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
  TensorDesc conf_trans_key_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_key_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_key_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_key, conf_trans_key_output_desc_y,
                         matmul_key, conf_transpose_key_perm, conf_transpose_key_shape);

  auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
  TensorDesc conf_trans_value_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_value_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_value_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_value, conf_trans_value_output_desc_y,
                         matmul_value, conf_transpose_value_perm, conf_transpose_value_shape);

  std::vector<Operator> inputs{input_x_data};
  std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVONNXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

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

TEST_F(attention_ln_qkv_onnx_fusion_test, attention_ln_qkv_onnx_fusion_layernorm_bmm_conftranspose_error) {
  // layernorm -> batchmatmulv2 -> conf_transpose pattern check failed
  ge::Graph graph("attention_ln_qkv_onnx_fusion_test4");

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 8;
  opti_compilation_info.soc_version = "soc_version";
  platform_info.ai_core_intrinsic_dtype_map["Intrinsic_vln"] = {"float32"};
  fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
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
  SetLayernorm(layer_norm, input_x_data, gamma, beta,
               input_desc_x, input_desc_gamma, input_desc_beta,
               output_desc_y, output_desc_mean, output_desc_variance);

  auto end_op_mean = op::Cast("end_op_mean");
  end_op_mean.set_input_x_by_name(layer_norm, "mean");
  end_op_mean.set_attr_dst_type(0);

  auto end_op_variance = op::Cast("end_op_variance");
  end_op_variance.set_input_x_by_name(layer_norm, "variance");
  end_op_variance.set_attr_dst_type(1);

  TensorDesc kernel_query_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor kernel_query_tensor(kernel_query_desc);
  auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
  TensorDesc kernel_key_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor kernel_key_tensor(kernel_key_desc);
  auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
  TensorDesc kernel_value_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor kernel_value_tensor(kernel_value_desc);
  auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);
  TensorDesc kernel_add_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor kernel_add_tensor(kernel_add_desc);
  auto kernel_add = op::Const("kernel_add").set_attr_value(kernel_add_tensor);

  TensorDesc bias_query_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
  Tensor bias_query_tensor(bias_query_desc);
  auto bias_query = op::Const("bias_query").set_attr_value(bias_query_tensor);
  TensorDesc bias_key_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
  Tensor bias_key_tensor(bias_key_desc);
  auto bias_key = op::Const("bias_key").set_attr_value(bias_key_tensor);
  TensorDesc bias_value_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
  Tensor bias_value_tensor(bias_value_desc);
  auto bias_value = op::Const("bias_value").set_attr_value(bias_value_tensor);
  TensorDesc bias_add_desc(ge::Shape({16, 64}), FORMAT_ND, DT_FLOAT16);
  Tensor bias_add_tensor(bias_add_desc);
  auto bias_add = op::Const("bias_add").set_attr_value(bias_add_tensor);

  auto matmul_add = op::BatchMatMulV2("matmul_add");
  TensorDesc mm_a_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  mm_a_output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
  SetMatmul(matmul_add, mm_a_output_desc_y, layer_norm, kernel_add, bias_add);

  TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor add_tensor(add_desc);
  auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
  auto add_0 = op::Add("add_0");
  TensorDesc add_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  add_0.update_output_desc_y(add_output_desc_y);
  add_0.set_input_x1(matmul_add);
  add_0.set_input_x2(add_const_op);

  auto matmul_query = op::BatchMatMulV2("matmul_query");
  TensorDesc mm_q_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_query, mm_q_output_desc_y, layer_norm, kernel_query, bias_query);

  auto matmul_key = op::BatchMatMulV2("matmul_key");
  TensorDesc mm_k_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_key, mm_k_output_desc_y, layer_norm, kernel_key, bias_key);

  auto matmul_value = op::BatchMatMulV2("matmul_value");
  TensorDesc mm_v_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_value, mm_v_output_desc_y, layer_norm, kernel_value, bias_value);

  auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
  TensorDesc conf_trans_query_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_query_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_query_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_query, conf_trans_query_output_desc_y,
                         matmul_query, conf_transpose_query_perm, conf_transpose_query_shape);

  auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
  TensorDesc conf_trans_key_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_key_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_key_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_key, conf_trans_key_output_desc_y,
                         matmul_key, conf_transpose_key_perm, conf_transpose_key_shape);

  auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
  TensorDesc conf_trans_value_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_value_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_value_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_value, conf_trans_value_output_desc_y,
                         matmul_value, conf_transpose_value_perm, conf_transpose_value_shape);

  std::vector<Operator> inputs{input_x_data};
  std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVONNXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

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

TEST_F(attention_ln_qkv_onnx_fusion_test, attention_ln_qkv_onnx_fusion_origin_shape_error) {
  // origin shape not aligned
  ge::Graph graph("attention_ln_qkv_onnx_fusion_test5");

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 8;
  opti_compilation_info.soc_version = "soc_version";
  platform_info.ai_core_intrinsic_dtype_map["Intrinsic_vln"] = {"float32"};
  fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
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
  SetLayernorm(layer_norm, input_x_data, gamma, beta,
               input_desc_x, input_desc_gamma, input_desc_beta,
               output_desc_y, output_desc_mean, output_desc_variance);

  auto end_op_mean = op::Cast("end_op_mean");
  end_op_mean.set_input_x_by_name(layer_norm, "mean");
  end_op_mean.set_attr_dst_type(0);

  auto end_op_variance = op::Cast("end_op_variance");
  end_op_variance.set_input_x_by_name(layer_norm, "variance");
  end_op_variance.set_attr_dst_type(1);

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

  auto matmul_query = op::BatchMatMulV2("matmul_query");
  TensorDesc mm_q_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  mm_q_output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
  SetMatmul(matmul_query, mm_q_output_desc_y, layer_norm, kernel_query, bias_query);

  auto matmul_key = op::BatchMatMulV2("matmul_key");
  TensorDesc mm_k_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_key, mm_k_output_desc_y, layer_norm, kernel_key, bias_key);

  auto matmul_value = op::BatchMatMulV2("matmul_value");
  TensorDesc mm_v_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_value, mm_v_output_desc_y, layer_norm, kernel_value, bias_value);

  auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
  TensorDesc conf_trans_query_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_query_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_query_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_query, conf_trans_query_output_desc_y,
                         matmul_query, conf_transpose_query_perm, conf_transpose_query_shape);

  auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
  TensorDesc conf_trans_key_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_key_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_key_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_key, conf_trans_key_output_desc_y,
                         matmul_key, conf_transpose_key_perm, conf_transpose_key_shape);

  auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
  TensorDesc conf_trans_value_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_value_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_value_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_value, conf_trans_value_output_desc_y,
                         matmul_value, conf_transpose_value_perm, conf_transpose_value_shape);

  TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor add_tensor(add_desc);
  auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
  auto add_0 = op::Add("add_0");
  TensorDesc add_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  add_0.update_output_desc_y(add_output_desc_y);
  add_0.set_input_x1_by_name(layer_norm, "y");
  add_0.set_input_x2(add_const_op);

  std::vector<Operator> inputs{input_x_data};
  std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVONNXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

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

TEST_F(attention_ln_qkv_onnx_fusion_test, attention_ln_qkv_onnx_fusion_conftranspose_outshpe_error) {
  // confusion_transpose out_shape check failed
  ge::Graph graph("attention_ln_qkv_onnx_fusion_test6");

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 8;
  opti_compilation_info.soc_version = "soc_version";
  platform_info.ai_core_intrinsic_dtype_map["Intrinsic_vln"] = {"float32"};
  fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
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
  SetLayernorm(layer_norm, input_x_data, gamma, beta,
               input_desc_x, input_desc_gamma, input_desc_beta,
               output_desc_y, output_desc_mean, output_desc_variance);

  auto end_op_mean = op::Cast("end_op_mean");
  end_op_mean.set_input_x_by_name(layer_norm, "mean");
  end_op_mean.set_attr_dst_type(0);

  auto end_op_variance = op::Cast("end_op_variance");
  end_op_variance.set_input_x_by_name(layer_norm, "variance");
  end_op_variance.set_attr_dst_type(1);

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

  auto matmul_query = op::BatchMatMulV2("matmul_query");
  TensorDesc mm_q_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  mm_q_output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
  SetMatmul(matmul_query, mm_q_output_desc_y, layer_norm, kernel_query, bias_query);

  auto matmul_key = op::BatchMatMulV2("matmul_key");
  TensorDesc mm_k_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_key, mm_k_output_desc_y, layer_norm, kernel_key, bias_key);

  auto matmul_value = op::BatchMatMulV2("matmul_value");
  TensorDesc mm_v_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_value, mm_v_output_desc_y, layer_norm, kernel_value, bias_value);

  auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
  TensorDesc conf_trans_query_output_desc_y(ge::Shape({24, 32, 16, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_query_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_query_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_query, conf_trans_query_output_desc_y,
                         matmul_query, conf_transpose_query_perm, conf_transpose_query_shape);

  auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
  TensorDesc conf_trans_key_output_desc_y(ge::Shape({24, 32, 16, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_key_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_key_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_key, conf_trans_key_output_desc_y,
                         matmul_key, conf_transpose_key_perm, conf_transpose_key_shape);

  auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
  TensorDesc conf_trans_value_output_desc_y(ge::Shape({24, 32, 16, 4, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_value_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_value_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_value, conf_trans_value_output_desc_y,
                         matmul_value, conf_transpose_value_perm, conf_transpose_value_shape);

  TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor add_tensor(add_desc);
  auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
  auto add_0 = op::Add("add_0");
  TensorDesc add_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  add_0.update_output_desc_y(add_output_desc_y);
  add_0.set_input_x1_by_name(layer_norm, "y");
  add_0.set_input_x2(add_const_op);

  std::vector<Operator> inputs{input_x_data};
  std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVONNXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

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

TEST_F(attention_ln_qkv_onnx_fusion_test, attention_ln_qkv_onnx_fusion_nshape_unsupported) {
  // unsupported n_shape
  ge::Graph graph("attention_ln_qkv_onnx_fusion_test7");

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 8;
  opti_compilation_info.soc_version = "soc_version";
  platform_info.ai_core_intrinsic_dtype_map["Intrinsic_vln"] = {"float32"};
  fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

  auto input_x_data = op::Data("input_x_data");
  TensorDesc input_x_desc(ge::Shape({32, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  input_x_data.update_input_desc_x(input_x_desc);
  input_x_data.update_output_desc_y(input_x_desc);

  TensorDesc gamma_desc(ge::Shape({512}), FORMAT_NHWC, DT_FLOAT16);
  Tensor gamma_tensor(gamma_desc);
  auto gamma = op::Const("gamma").set_attr_value(gamma_tensor);

  TensorDesc beta_desc(ge::Shape({512}), FORMAT_NHWC, DT_FLOAT16);
  Tensor beta_tensor(beta_desc);
  auto beta = op::Const("beta").set_attr_value(beta_tensor);

  auto layer_norm = op::LayerNorm("layer_norm_0");
  TensorDesc input_desc_x(ge::Shape({32, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  TensorDesc input_desc_gamma(ge::Shape({512}), FORMAT_NHWC, DT_FLOAT16);
  TensorDesc input_desc_beta(ge::Shape({512}), FORMAT_NHWC, DT_FLOAT16);
  TensorDesc output_desc_y(ge::Shape({32, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  output_desc_y.SetOriginShape(ge::Shape({12288, 512}));
  TensorDesc output_desc_mean(ge::Shape({512}), FORMAT_NHWC, DT_FLOAT16);
  TensorDesc output_desc_variance(ge::Shape({512}), FORMAT_NHWC, DT_FLOAT16);
  SetLayernorm(layer_norm, input_x_data, gamma, beta,
               input_desc_x, input_desc_gamma, input_desc_beta,
               output_desc_y, output_desc_mean, output_desc_variance);

  auto end_op_mean = op::Cast("end_op_mean");
  end_op_mean.set_input_x_by_name(layer_norm, "mean");
  end_op_mean.set_attr_dst_type(0);

  auto end_op_variance = op::Cast("end_op_variance");
  end_op_variance.set_input_x_by_name(layer_norm, "variance");
  end_op_variance.set_attr_dst_type(1);

  TensorDesc kernel_query_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor kernel_query_tensor(kernel_query_desc);
  auto kernel_query = op::Const("kernel_query").set_attr_value(kernel_query_tensor);
  TensorDesc kernel_key_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor kernel_key_tensor(kernel_key_desc);
  auto kernel_key = op::Const("kernel_key").set_attr_value(kernel_key_tensor);
  TensorDesc kernel_value_desc(ge::Shape({32, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor kernel_value_tensor(kernel_value_desc);
  auto kernel_value = op::Const("kernel_value").set_attr_value(kernel_value_tensor);

  TensorDesc bias_query_desc(ge::Shape({16, 32}), FORMAT_ND, DT_FLOAT16);
  Tensor bias_query_tensor(bias_query_desc);
  auto bias_query = op::Const("bias_query").set_attr_value(bias_query_tensor);
  TensorDesc bias_key_desc(ge::Shape({16, 32}), FORMAT_ND, DT_FLOAT16);
  Tensor bias_key_tensor(bias_key_desc);
  auto bias_key = op::Const("bias_key").set_attr_value(bias_key_tensor);
  TensorDesc bias_value_desc(ge::Shape({16, 32}), FORMAT_ND, DT_FLOAT16);
  Tensor bias_value_tensor(bias_value_desc);
  auto bias_value = op::Const("bias_value").set_attr_value(bias_value_tensor);

  auto matmul_query = op::BatchMatMulV2("matmul_query");
  TensorDesc mm_q_output_desc_y(ge::Shape({32, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  mm_q_output_desc_y.SetOriginShape(ge::Shape({12288, 512}));
  SetMatmul(matmul_query, mm_q_output_desc_y, layer_norm, kernel_query, bias_query);

  auto matmul_key = op::BatchMatMulV2("matmul_key");
  TensorDesc mm_k_output_desc_y(ge::Shape({32, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_key, mm_k_output_desc_y, layer_norm, kernel_key, bias_key);

  auto matmul_value = op::BatchMatMulV2("matmul_value");
  TensorDesc mm_v_output_desc_y(ge::Shape({32, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_value, mm_v_output_desc_y, layer_norm, kernel_value, bias_value);

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

  TensorDesc add_desc(ge::Shape({32, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor add_tensor(add_desc);
  auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
  auto add_0 = op::Add("add_0");
  TensorDesc add_output_desc_y(ge::Shape({32, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  add_0.update_output_desc_y(add_output_desc_y);
  add_0.set_input_x1_by_name(layer_norm, "y");
  add_0.set_input_x2(add_const_op);

  std::vector<Operator> inputs{input_x_data};
  std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVONNXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{24, 8, 4, 32, 16, 16};
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

TEST_F(attention_ln_qkv_onnx_fusion_test, attention_ln_qkv_onnx_fusion_mshape_unspported) {
  // m_shape less than 1536
  ge::Graph graph("attention_ln_qkv_onnx_fusion_test8");

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 8;
  opti_compilation_info.soc_version = "soc_version";
  platform_info.ai_core_intrinsic_dtype_map["Intrinsic_vln"] = {"float32"};
  fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

  auto input_x_data = op::Data("input_x_data");
  TensorDesc input_x_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  input_x_data.update_input_desc_x(input_x_desc);
  input_x_data.update_output_desc_y(input_x_desc);

  TensorDesc gamma_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
  Tensor gamma_tensor(gamma_desc);
  auto gamma = op::Const("gamma").set_attr_value(gamma_tensor);

  TensorDesc beta_desc(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
  Tensor beta_tensor(beta_desc);
  auto beta = op::Const("beta").set_attr_value(beta_tensor);

  auto layer_norm = op::LayerNorm("layer_norm_0");
  TensorDesc input_desc_x(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  TensorDesc input_desc_gamma(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
  TensorDesc input_desc_beta(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
  TensorDesc output_desc_y(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  output_desc_y.SetOriginShape(ge::Shape({1024, 1024}));
  TensorDesc output_desc_mean(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
  TensorDesc output_desc_variance(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
  SetLayernorm(layer_norm, input_x_data, gamma, beta,
               input_desc_x, input_desc_gamma, input_desc_beta,
               output_desc_y, output_desc_mean, output_desc_variance);

  auto end_op_mean = op::Cast("end_op_mean");
  end_op_mean.set_input_x_by_name(layer_norm, "mean");
  end_op_mean.set_attr_dst_type(0);

  auto end_op_variance = op::Cast("end_op_variance");
  end_op_variance.set_input_x_by_name(layer_norm, "variance");
  end_op_variance.set_attr_dst_type(1);

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

  auto matmul_query = op::BatchMatMulV2("matmul_query");
  TensorDesc mm_q_output_desc_y(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  mm_q_output_desc_y.SetOriginShape(ge::Shape({1024, 1024}));
  SetMatmul(matmul_query, mm_q_output_desc_y, layer_norm, kernel_query, bias_query);

  auto matmul_key = op::BatchMatMulV2("matmul_key");
  TensorDesc mm_k_output_desc_y(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_key, mm_k_output_desc_y, layer_norm, kernel_key, bias_key);

  auto matmul_value = op::BatchMatMulV2("matmul_value");
  TensorDesc mm_v_output_desc_y(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_value, mm_v_output_desc_y, layer_norm, kernel_value, bias_value);

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

  TensorDesc add_desc(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor add_tensor(add_desc);
  auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
  auto add_0 = op::Add("add_0");
  TensorDesc add_output_desc_y(ge::Shape({64, 64, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  add_0.update_output_desc_y(add_output_desc_y);
  add_0.set_input_x1_by_name(layer_norm, "y");
  add_0.set_input_x2(add_const_op);

  std::vector<Operator> inputs{input_x_data};
  std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVONNXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{2, 16, 4, 32, 16, 16};
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

TEST_F(attention_ln_qkv_onnx_fusion_test, attention_ln_qkv_onnx_fusion_dtype_unsupported) {
  // unsupported dtype
  ge::Graph graph("attention_ln_qkv_onnx_fusion_test9");

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 8;
  opti_compilation_info.soc_version = "soc_version";
  platform_info.ai_core_intrinsic_dtype_map["Intrinsic_vln"] = {"float32"};
  fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
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
  TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT);
  output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
  TensorDesc output_desc_mean(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
  TensorDesc output_desc_variance(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
  SetLayernorm(layer_norm, input_x_data, gamma, beta,
               input_desc_x, input_desc_gamma, input_desc_beta,
               output_desc_y, output_desc_mean, output_desc_variance);

  auto end_op_mean = op::Cast("end_op_mean");
  end_op_mean.set_input_x_by_name(layer_norm, "mean");
  end_op_mean.set_attr_dst_type(0);

  auto end_op_variance = op::Cast("end_op_variance");
  end_op_variance.set_input_x_by_name(layer_norm, "variance");
  end_op_variance.set_attr_dst_type(1);

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

  auto matmul_query = op::BatchMatMulV2("matmul_query");
  TensorDesc mm_q_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  mm_q_output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
  SetMatmul(matmul_query, mm_q_output_desc_y, layer_norm, kernel_query, bias_query);

  auto matmul_key = op::BatchMatMulV2("matmul_key");
  TensorDesc mm_k_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_key, mm_k_output_desc_y, layer_norm, kernel_key, bias_key);

  auto matmul_value = op::BatchMatMulV2("matmul_value");
  TensorDesc mm_v_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_value, mm_v_output_desc_y, layer_norm, kernel_value, bias_value);

  auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
  TensorDesc conf_trans_query_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_query_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_query_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_query, conf_trans_query_output_desc_y,
                         matmul_query, conf_transpose_query_perm, conf_transpose_query_shape);

  auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
  TensorDesc conf_trans_key_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_key_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_key_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_key, conf_trans_key_output_desc_y,
                         matmul_key, conf_transpose_key_perm, conf_transpose_key_shape);

  auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
  TensorDesc conf_trans_value_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_value_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_value_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_value, conf_trans_value_output_desc_y,
                         matmul_value, conf_transpose_value_perm, conf_transpose_value_shape);

  TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor add_tensor(add_desc);
  auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
  auto add_0 = op::Add("add_0");
  TensorDesc add_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  add_0.update_output_desc_y(add_output_desc_y);
  add_0.set_input_x1_by_name(layer_norm, "y");
  add_0.set_input_x2(add_const_op);

  std::vector<Operator> inputs{input_x_data};
  std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVONNXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

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

TEST_F(attention_ln_qkv_onnx_fusion_test, attention_ln_qkv_onnx_fusion_format_unsupported) {
  // unsupported format
  ge::Graph graph("attention_ln_qkv_onnx_fusion_test10");

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 8;
  opti_compilation_info.soc_version = "soc_version";
  platform_info.ai_core_intrinsic_dtype_map["Intrinsic_vln"] = {"float32"};
  fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
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
  TensorDesc output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_NHWC, DT_FLOAT16);
  output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
  TensorDesc output_desc_mean(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
  TensorDesc output_desc_variance(ge::Shape({1024}), FORMAT_NHWC, DT_FLOAT16);
  SetLayernorm(layer_norm, input_x_data, gamma, beta,
               input_desc_x, input_desc_gamma, input_desc_beta,
               output_desc_y, output_desc_mean, output_desc_variance);

  auto end_op_mean = op::Cast("end_op_mean");
  end_op_mean.set_input_x_by_name(layer_norm, "mean");
  end_op_mean.set_attr_dst_type(0);

  auto end_op_variance = op::Cast("end_op_variance");
  end_op_variance.set_input_x_by_name(layer_norm, "variance");
  end_op_variance.set_attr_dst_type(1);

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

  auto matmul_query = op::BatchMatMulV2("matmul_query");
  TensorDesc mm_q_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  mm_q_output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
  SetMatmul(matmul_query, mm_q_output_desc_y, layer_norm, kernel_query, bias_query);

  auto matmul_key = op::BatchMatMulV2("matmul_key");
  TensorDesc mm_k_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_key, mm_k_output_desc_y, layer_norm, kernel_key, bias_key);

  auto matmul_value = op::BatchMatMulV2("matmul_value");
  TensorDesc mm_v_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_value, mm_v_output_desc_y, layer_norm, kernel_value, bias_value);

  auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
  TensorDesc conf_trans_query_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_query_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_query_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_query, conf_trans_query_output_desc_y,
                         matmul_query, conf_transpose_query_perm, conf_transpose_query_shape);

  auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
  TensorDesc conf_trans_key_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_key_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_key_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_key, conf_trans_key_output_desc_y,
                         matmul_key, conf_transpose_key_perm, conf_transpose_key_shape);

  auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
  TensorDesc conf_trans_value_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_value_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_value_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_value, conf_trans_value_output_desc_y,
                         matmul_value, conf_transpose_value_perm, conf_transpose_value_shape);

  TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor add_tensor(add_desc);
  auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
  auto add_0 = op::Add("add_0");
  TensorDesc add_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  add_0.update_output_desc_y(add_output_desc_y);
  add_0.set_input_x1_by_name(layer_norm, "y");
  add_0.set_input_x2(add_const_op);

  std::vector<Operator> inputs{input_x_data};
  std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVONNXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

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

TEST_F(attention_ln_qkv_onnx_fusion_test, attention_ln_qkv_onnx_fusion_platform_error) {
  // platform check failed
  ge::Graph graph("attention_ln_qkv_onnx_fusion_test11");

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 2;
  opti_compilation_info.soc_version = "soc_version";
  fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
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
  SetLayernorm(layer_norm, input_x_data, gamma, beta,
               input_desc_x, input_desc_gamma, input_desc_beta,
               output_desc_y, output_desc_mean, output_desc_variance);

  auto end_op_mean = op::Cast("end_op_mean");
  end_op_mean.set_input_x_by_name(layer_norm, "mean");
  end_op_mean.set_attr_dst_type(0);

  auto end_op_variance = op::Cast("end_op_variance");
  end_op_variance.set_input_x_by_name(layer_norm, "variance");
  end_op_variance.set_attr_dst_type(1);

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

  auto matmul_query = op::BatchMatMulV2("matmul_query");
  TensorDesc mm_q_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  mm_q_output_desc_y.SetOriginShape(ge::Shape({12288, 1024}));
  SetMatmul(matmul_query, mm_q_output_desc_y, layer_norm, kernel_query, bias_query);

  auto matmul_key = op::BatchMatMulV2("matmul_key");
  TensorDesc mm_k_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_key, mm_k_output_desc_y, layer_norm, kernel_key, bias_key);

  auto matmul_value = op::BatchMatMulV2("matmul_value");
  TensorDesc mm_v_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  SetMatmul(matmul_value, mm_v_output_desc_y, layer_norm, kernel_value, bias_value);

  auto conf_transpose_query = op::ConfusionTransposeD("conf_transpose_query");
  TensorDesc conf_trans_query_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_query_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_query_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_query, conf_trans_query_output_desc_y,
                         matmul_query, conf_transpose_query_perm, conf_transpose_query_shape);

  auto conf_transpose_key = op::ConfusionTransposeD("conf_transpose_key");
  TensorDesc conf_trans_key_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_key_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_key_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_key, conf_trans_key_output_desc_y,
                         matmul_key, conf_transpose_key_perm, conf_transpose_key_shape);

  auto conf_transpose_value = op::ConfusionTransposeD("conf_transpose_value");
  TensorDesc conf_trans_value_output_desc_y(ge::Shape({24, 16, 4, 32, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::Operator::OpListInt conf_transpose_value_perm = {0, 2, 1, 3};
  ge::Operator::OpListInt conf_transpose_value_shape = {24, 512, 16, 64};
  SetConfusionTransposeD(conf_transpose_value, conf_trans_value_output_desc_y,
                         matmul_value, conf_transpose_value_perm, conf_transpose_value_shape);

  TensorDesc add_desc(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Tensor add_tensor(add_desc);
  auto add_const_op = op::Const("add_const_op").set_attr_value(add_tensor);
  auto add_0 = op::Add("add_0");
  TensorDesc add_output_desc_y(ge::Shape({64, 768, 16, 16}), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  add_0.update_output_desc_y(add_output_desc_y);
  add_0.set_input_x1_by_name(layer_norm, "y");
  add_0.set_input_x2(add_const_op);

  std::vector<Operator> inputs{input_x_data};
  std::vector<Operator> outputs{layer_norm, conf_transpose_query, conf_transpose_key, conf_transpose_value};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionLnQKVONNXFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

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
