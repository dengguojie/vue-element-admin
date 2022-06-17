#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_norm_ops.h"
#include "array_ops.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"
#define private public
#include "fusion_pass_test_utils.h"
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;

class ln_dropout_grad_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "ln_dropout_grad_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "ln_dropout_grad_fusion_pass_test TearDown" << std::endl;
    }

};

TEST_F(ln_dropout_grad_fusion_pass_test, ln_dropout_grad_fusion_pass_test_1) {

  ge::Graph graph("ln_dropout_grad_fusion_pass_test_1");
  auto input_data = op::Data("input_data");
  std::vector<int64_t> dims{3, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_FLOAT16);
  input_data.update_input_desc_x(tensorDesc);
  input_data.update_output_desc_y(tensorDesc);

  auto input_data_dy = op::Data("input_data_dy");
  std::vector<int64_t> dims_dy{3, 32};
  ge::Shape shape_dy(dims_dy);
  ge::TensorDesc tensorDescDy(shape_dy, FORMAT_ND, DT_FLOAT16);
  input_data_dy.update_input_desc_x(tensorDescDy);
  input_data_dy.update_output_desc_y(tensorDescDy);

  auto input_data_mean = op::Data("input_data_mean");
  std::vector<int64_t> dims_mean{3, 1};
  ge::Shape shape_mean(dims_mean);
  ge::TensorDesc tensorDescMean(shape_mean, FORMAT_ND, DT_FLOAT16);
  input_data_mean.update_input_desc_x(tensorDescMean);
  input_data_mean.update_output_desc_y(tensorDescMean);

  auto input_data_variance = op::Data("input_data_variance");
  std::vector<int64_t> dims_variance{3, 1};
  ge::Shape shape_variance(dims_variance);
  ge::TensorDesc tensorDescVariance(shape_variance, FORMAT_ND, DT_FLOAT16);
  input_data_variance.update_input_desc_x(tensorDescVariance);
  input_data_variance.update_output_desc_y(tensorDescVariance);

  auto input_data_gamma = op::Data("input_data_gamma");
  std::vector<int64_t> dims_gamma{32};
  ge::Shape shape_gamma(dims_gamma);
  ge::TensorDesc tensorDescGamma(shape_gamma, FORMAT_ND, DT_FLOAT16);
  input_data_gamma.update_input_desc_x(tensorDescGamma);
  input_data_gamma.update_output_desc_y(tensorDescGamma);

  auto layer_norm_grad_op = op::LayerNormGrad("layer_norm_grad_0");
  layer_norm_grad_op.set_input_x(input_data);
  layer_norm_grad_op.set_input_dy(input_data_dy);
  layer_norm_grad_op.set_input_mean(input_data_mean);
  layer_norm_grad_op.set_input_variance(input_data_variance);
  layer_norm_grad_op.set_input_gamma(input_data_gamma);

  auto input_data_mask = op::Data("input_mask");
  std::vector<int64_t> dims_mask{3, 32};
  ge::Shape shape_mask(dims_mask);
  ge::TensorDesc tensorDescMask(shape_mask, FORMAT_ND, DT_UINT8);
  input_data_mask.update_input_desc_x(tensorDescMask);
  input_data_mask.update_output_desc_y(tensorDescMask);

  auto dropout_domask_op = op::DropOutDoMaskV3D("dropout_domask_v3d_0");
  dropout_domask_op.set_input_x(layer_norm_grad_op, "pd_x");
  dropout_domask_op.set_input_mask(input_data_mask);
  dropout_domask_op.set_attr_keep_prob(0.5f);

  auto end_op_pd_x = op::Cast("const0_op");
  end_op_pd_x.set_input_x(layer_norm_grad_op, "pd_x")
             .set_attr_dst_type(0);

  auto end_op_pd_x_dropout = op::Cast("const1_op");
  end_op_pd_x_dropout.set_input_x(dropout_domask_op, "y")
                   .set_attr_dst_type(0);

  auto end_op_pd_gamma = op::Cast("const2_op");
  end_op_pd_gamma.set_input_x(layer_norm_grad_op, "pd_gamma")
                 .set_attr_dst_type(0);

  auto end_op_pd_beta = op::Cast("const3_op");
  end_op_pd_beta.set_input_x(layer_norm_grad_op, "pd_beta")
                .set_attr_dst_type(1);

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 32;
  opti_compilation_info.soc_version = "Ascend910A";
  platform_info.str_info.short_soc_version = "Ascend910";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platform_info;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

  std::vector<Operator> inputs{input_data, input_data_dy, input_data_mean, input_data_variance,
                                 input_data_gamma, input_data_mask};
  std::vector<Operator> outputs{end_op_pd_x, end_op_pd_x_dropout, end_op_pd_gamma, end_op_pd_beta};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormGradDropOutDoMaskV3DFusionPass",
                                              fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
  std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
  fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
  fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradDropOutDoMaskV3DFusionPass"].GetMatchTimes(), 1);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradDropOutDoMaskV3DFusionPass"].GetEffectTimes(), 1);
}

TEST_F(ln_dropout_grad_fusion_pass_test, ln_dropout_grad_fusion_pass_test_2) {
  ge::Graph graph("ln_dropout_grad_fusion_pass_test_2");
  auto input_data = op::Data("input_data");
  std::vector<int64_t> dims{3, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_FLOAT16);
  input_data.update_input_desc_x(tensorDesc);
  input_data.update_output_desc_y(tensorDesc);

  auto input_data_dy = op::Data("input_data_dy");
  std::vector<int64_t> dims_dy{3, 32};
  ge::Shape shape_dy(dims_dy);
  ge::TensorDesc tensorDescDy(shape_dy, FORMAT_ND, DT_FLOAT16);
  input_data_dy.update_input_desc_x(tensorDescDy);
  input_data_dy.update_output_desc_y(tensorDescDy);

  auto input_data_mean = op::Data("input_data_mean");
  std::vector<int64_t> dims_mean{3, 1};
  ge::Shape shape_mean(dims_mean);
  ge::TensorDesc tensorDescMean(shape_mean, FORMAT_ND, DT_FLOAT16);
  input_data_mean.update_input_desc_x(tensorDescMean);
  input_data_mean.update_output_desc_y(tensorDescMean);

  auto input_data_variance = op::Data("input_data_variance");
  std::vector<int64_t> dims_variance{3, 1};
  ge::Shape shape_variance(dims_variance);
  ge::TensorDesc tensorDescVariance(shape_variance, FORMAT_ND, DT_FLOAT16);
  input_data_variance.update_input_desc_x(tensorDescVariance);
  input_data_variance.update_output_desc_y(tensorDescVariance);

  auto input_data_gamma = op::Data("input_data_gamma");
  std::vector<int64_t> dims_gamma{32};
  ge::Shape shape_gamma(dims_gamma);
  ge::TensorDesc tensorDescGamma(shape_gamma, FORMAT_ND, DT_FLOAT16);
  input_data_gamma.update_input_desc_x(tensorDescGamma);
  input_data_gamma.update_output_desc_y(tensorDescGamma);

  auto layer_norm_grad_op = op::LayerNormGrad("layer_norm_grad_0");
  layer_norm_grad_op.set_input_x(input_data);
  layer_norm_grad_op.set_input_dy(input_data_dy);
  layer_norm_grad_op.set_input_mean(input_data_mean);
  layer_norm_grad_op.set_input_variance(input_data_variance);
  layer_norm_grad_op.set_input_gamma(input_data_gamma);

  auto mid_cast_op = op::Cast("mid_cast");
  mid_cast_op.set_input_x(layer_norm_grad_op, "pd_x")
             .set_attr_dst_type(0);

  auto input_data_mask = op::Data("input_mask");
  std::vector<int64_t> dims_mask{3, 32};
  ge::Shape shape_mask(dims_mask);
  ge::TensorDesc tensorDescMask(shape_mask, FORMAT_ND, DT_UINT8);
  input_data_mask.update_input_desc_x(tensorDescMask);
  input_data_mask.update_output_desc_y(tensorDescMask);

  auto dropout_domask_op = op::DropOutDoMaskV3D("dropout_domask_v3d_0");
  dropout_domask_op.set_input_x(mid_cast_op, "y");
  dropout_domask_op.set_input_mask(input_data_mask);
  dropout_domask_op.set_attr_keep_prob(0.5f);

  auto end_op_pd_x = op::Cast("const0_op");
  end_op_pd_x.set_input_x(mid_cast_op, "y")
  .set_attr_dst_type(1);

  auto end_op_pd_x_dropout = op::Cast("const1_op");
  end_op_pd_x_dropout.set_input_x(dropout_domask_op, "y")
  .set_attr_dst_type(1);

  auto end_op_pd_gamma = op::Cast("const2_op");
  end_op_pd_gamma.set_input_x(layer_norm_grad_op, "pd_gamma")
  .set_attr_dst_type(1);

  auto end_op_pd_beta = op::Cast("const3_op");
  end_op_pd_beta.set_input_x(layer_norm_grad_op, "pd_beta")
  .set_attr_dst_type(1);

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 32;
  opti_compilation_info.soc_version = "Ascend910A";
  platform_info.str_info.short_soc_version = "Ascend910";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platform_info;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

  std::vector<Operator> inputs{input_data, input_data_dy, input_data_mean, input_data_variance,
                               input_data_gamma, input_data_mask};
  std::vector<Operator> outputs{end_op_pd_x, end_op_pd_x_dropout, end_op_pd_gamma, end_op_pd_beta};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormGradDropOutDoMaskV3DFusionPass",
  fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
  std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
  fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
  fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradDropOutDoMaskV3DFusionPass"].GetMatchTimes(), 1);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradDropOutDoMaskV3DFusionPass"].GetEffectTimes(), 1);
}

TEST_F(ln_dropout_grad_fusion_pass_test, ln_dropout_grad_fusion_pass_test_3) {
  ge::Graph graph("ln_dropout_grad_fusion_pass_test_invalid_gamma_shape");
  auto input_data = op::Data("input_data");
  std::vector<int64_t> dims{3, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_FLOAT16);
  input_data.update_input_desc_x(tensorDesc);
  input_data.update_output_desc_y(tensorDesc);

  auto input_data_dy = op::Data("input_data_dy");
  std::vector<int64_t> dims_dy{3, 32};
  ge::Shape shape_dy(dims_dy);
  ge::TensorDesc tensorDescDy(shape_dy, FORMAT_ND, DT_FLOAT16);
  input_data_dy.update_input_desc_x(tensorDescDy);
  input_data_dy.update_output_desc_y(tensorDescDy);

  auto input_data_mean = op::Data("input_data_mean");
  std::vector<int64_t> dims_mean{3, 1};
  ge::Shape shape_mean(dims_mean);
  ge::TensorDesc tensorDescMean(shape_mean, FORMAT_ND, DT_FLOAT16);
  input_data_mean.update_input_desc_x(tensorDescMean);
  input_data_mean.update_output_desc_y(tensorDescMean);

  auto input_data_variance = op::Data("input_data_variance");
  std::vector<int64_t> dims_variance{3, 1};
  ge::Shape shape_variance(dims_variance);
  ge::TensorDesc tensorDescVariance(shape_variance, FORMAT_ND, DT_FLOAT16);
  input_data_variance.update_input_desc_x(tensorDescVariance);
  input_data_variance.update_output_desc_y(tensorDescVariance);

  auto input_data_gamma = op::Data("input_data_gamma");
  std::vector<int64_t> dims_gamma{32, 5};
  ge::Shape shape_gamma(dims_gamma);
  ge::TensorDesc tensorDescGamma(shape_gamma, FORMAT_ND, DT_FLOAT16);
  input_data_gamma.update_input_desc_x(tensorDescGamma);
  input_data_gamma.update_output_desc_y(tensorDescGamma);

  auto layer_norm_grad_op = op::LayerNormGrad("layer_norm_grad_0");
  layer_norm_grad_op.set_input_x(input_data);
  layer_norm_grad_op.set_input_dy(input_data_dy);
  layer_norm_grad_op.set_input_mean(input_data_mean);
  layer_norm_grad_op.set_input_variance(input_data_variance);
  layer_norm_grad_op.set_input_gamma(input_data_gamma);

  auto mid_cast_op = op::Cast("mid_cast");
  mid_cast_op.set_input_x(layer_norm_grad_op, "pd_x")
  .set_attr_dst_type(0);

  auto input_data_mask = op::Data("input_mask");
  std::vector<int64_t> dims_mask{3, 32};
  ge::Shape shape_mask(dims_mask);
  ge::TensorDesc tensorDescMask(shape_mask, FORMAT_ND, DT_UINT8);
  input_data_mask.update_input_desc_x(tensorDescMask);
  input_data_mask.update_output_desc_y(tensorDescMask);

  auto dropout_domask_op = op::DropOutDoMaskV3D("dropout_domask_v3d_0");
  dropout_domask_op.set_input_x(mid_cast_op, "y");
  dropout_domask_op.set_input_mask(input_data_mask);
  dropout_domask_op.set_attr_keep_prob(0.5f);

  auto end_op_pd_x = op::Cast("const0_op");
  end_op_pd_x.set_input_x(mid_cast_op, "y")
  .set_attr_dst_type(1);

  auto end_op_pd_x_dropout = op::Cast("const1_op");
  end_op_pd_x_dropout.set_input_x(dropout_domask_op, "y")
  .set_attr_dst_type(1);

  auto end_op_pd_gamma = op::Cast("const2_op");
  end_op_pd_gamma.set_input_x(layer_norm_grad_op, "pd_gamma")
  .set_attr_dst_type(1);

  auto end_op_pd_beta = op::Cast("const3_op");
  end_op_pd_beta.set_input_x(layer_norm_grad_op, "pd_beta")
  .set_attr_dst_type(1);

  std::vector<Operator> inputs{input_data, input_data_dy, input_data_mean, input_data_variance,
                               input_data_gamma, input_data_mask};
  std::vector<Operator> outputs{end_op_pd_x, end_op_pd_x_dropout, end_op_pd_gamma, end_op_pd_beta};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormGradDropOutDoMaskV3DFusionPass",
  fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
  std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
  fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
  fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradDropOutDoMaskV3DFusionPass"].GetMatchTimes(), 1);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradDropOutDoMaskV3DFusionPass"].GetEffectTimes(), 0);
}

TEST_F(ln_dropout_grad_fusion_pass_test, ln_dropout_grad_fusion_pass_test_4) {
  ge::Graph graph("ln_dropout_grad_fusion_pass_test_not_align");
  auto input_data = op::Data("input_data");
  std::vector<int64_t> dims{3, 31};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_FLOAT16);
  input_data.update_input_desc_x(tensorDesc);
  input_data.update_output_desc_y(tensorDesc);

  auto input_data_dy = op::Data("input_data_dy");
  std::vector<int64_t> dims_dy{3, 31};
  ge::Shape shape_dy(dims_dy);
  ge::TensorDesc tensorDescDy(shape_dy, FORMAT_ND, DT_FLOAT16);
  input_data_dy.update_input_desc_x(tensorDescDy);
  input_data_dy.update_output_desc_y(tensorDescDy);

  auto input_data_mean = op::Data("input_data_mean");
  std::vector<int64_t> dims_mean{3, 1};
  ge::Shape shape_mean(dims_mean);
  ge::TensorDesc tensorDescMean(shape_mean, FORMAT_ND, DT_FLOAT16);
  input_data_mean.update_input_desc_x(tensorDescMean);
  input_data_mean.update_output_desc_y(tensorDescMean);

  auto input_data_variance = op::Data("input_data_variance");
  std::vector<int64_t> dims_variance{3, 1};
  ge::Shape shape_variance(dims_variance);
  ge::TensorDesc tensorDescVariance(shape_variance, FORMAT_ND, DT_FLOAT16);
  input_data_variance.update_input_desc_x(tensorDescVariance);
  input_data_variance.update_output_desc_y(tensorDescVariance);

  auto input_data_gamma = op::Data("input_data_gamma");
  std::vector<int64_t> dims_gamma{31};
  ge::Shape shape_gamma(dims_gamma);
  ge::TensorDesc tensorDescGamma(shape_gamma, FORMAT_ND, DT_FLOAT16);
  input_data_gamma.update_input_desc_x(tensorDescGamma);
  input_data_gamma.update_output_desc_y(tensorDescGamma);

  auto layer_norm_grad_op = op::LayerNormGrad("layer_norm_grad_0");
  layer_norm_grad_op.set_input_x(input_data);
  layer_norm_grad_op.set_input_dy(input_data_dy);
  layer_norm_grad_op.set_input_mean(input_data_mean);
  layer_norm_grad_op.set_input_variance(input_data_variance);
  layer_norm_grad_op.set_input_gamma(input_data_gamma);

  auto mid_cast_op = op::Cast("mid_cast");
  mid_cast_op.set_input_x(layer_norm_grad_op, "pd_x")
  .set_attr_dst_type(0);

  auto input_data_mask = op::Data("input_mask");
  std::vector<int64_t> dims_mask{3, 31};
  ge::Shape shape_mask(dims_mask);
  ge::TensorDesc tensorDescMask(shape_mask, FORMAT_ND, DT_UINT8);
  input_data_mask.update_input_desc_x(tensorDescMask);
  input_data_mask.update_output_desc_y(tensorDescMask);

  auto dropout_domask_op = op::DropOutDoMaskV3D("dropout_domask_v3d_0");
  dropout_domask_op.set_input_x(mid_cast_op, "y");
  dropout_domask_op.set_input_mask(input_data_mask);
  dropout_domask_op.set_attr_keep_prob(0.5f);

  auto end_op_pd_x = op::Cast("const0_op");
  end_op_pd_x.set_input_x(mid_cast_op, "y")
  .set_attr_dst_type(1);

  auto end_op_pd_x_dropout = op::Cast("const1_op");
  end_op_pd_x_dropout.set_input_x(dropout_domask_op, "y")
  .set_attr_dst_type(1);

  auto end_op_pd_gamma = op::Cast("const2_op");
  end_op_pd_gamma.set_input_x(layer_norm_grad_op, "pd_gamma")
  .set_attr_dst_type(1);

  auto end_op_pd_beta = op::Cast("const3_op");
  end_op_pd_beta.set_input_x(layer_norm_grad_op, "pd_beta")
  .set_attr_dst_type(1);

  std::vector<Operator> inputs{input_data, input_data_dy, input_data_mean, input_data_variance,
                               input_data_gamma, input_data_mask};
  std::vector<Operator> outputs{end_op_pd_x, end_op_pd_x_dropout, end_op_pd_gamma, end_op_pd_beta};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormGradDropOutDoMaskV3DFusionPass",
  fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
  std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
  fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
  fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradDropOutDoMaskV3DFusionPass"].GetMatchTimes(), 1);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradDropOutDoMaskV3DFusionPass"].GetEffectTimes(), 0);
}

TEST_F(ln_dropout_grad_fusion_pass_test, ln_dropout_grad_fusion_pass_test_5) {
  ge::Graph graph("ln_dropout_grad_fusion_pass_test_big_axis");
  auto input_data = op::Data("input_data");
  std::vector<int64_t> dims{3, 2048};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_FLOAT16);
  input_data.update_input_desc_x(tensorDesc);
  input_data.update_output_desc_y(tensorDesc);

  auto input_data_dy = op::Data("input_data_dy");
  std::vector<int64_t> dims_dy{3, 2048};
  ge::Shape shape_dy(dims_dy);
  ge::TensorDesc tensorDescDy(shape_dy, FORMAT_ND, DT_FLOAT16);
  input_data_dy.update_input_desc_x(tensorDescDy);
  input_data_dy.update_output_desc_y(tensorDescDy);

  auto input_data_mean = op::Data("input_data_mean");
  std::vector<int64_t> dims_mean{3, 1};
  ge::Shape shape_mean(dims_mean);
  ge::TensorDesc tensorDescMean(shape_mean, FORMAT_ND, DT_FLOAT16);
  input_data_mean.update_input_desc_x(tensorDescMean);
  input_data_mean.update_output_desc_y(tensorDescMean);

  auto input_data_variance = op::Data("input_data_variance");
  std::vector<int64_t> dims_variance{3, 1};
  ge::Shape shape_variance(dims_variance);
  ge::TensorDesc tensorDescVariance(shape_variance, FORMAT_ND, DT_FLOAT16);
  input_data_variance.update_input_desc_x(tensorDescVariance);
  input_data_variance.update_output_desc_y(tensorDescVariance);

  auto input_data_gamma = op::Data("input_data_gamma");
  std::vector<int64_t> dims_gamma{2048};
  ge::Shape shape_gamma(dims_gamma);
  ge::TensorDesc tensorDescGamma(shape_gamma, FORMAT_ND, DT_FLOAT16);
  input_data_gamma.update_input_desc_x(tensorDescGamma);
  input_data_gamma.update_output_desc_y(tensorDescGamma);

  auto layer_norm_grad_op = op::LayerNormGrad("layer_norm_grad_0");
  layer_norm_grad_op.set_input_x(input_data);
  layer_norm_grad_op.set_input_dy(input_data_dy);
  layer_norm_grad_op.set_input_mean(input_data_mean);
  layer_norm_grad_op.set_input_variance(input_data_variance);
  layer_norm_grad_op.set_input_gamma(input_data_gamma);

  auto mid_cast_op = op::Cast("mid_cast");
  mid_cast_op.set_input_x(layer_norm_grad_op, "pd_x")
  .set_attr_dst_type(0);

  auto input_data_mask = op::Data("input_mask");
  std::vector<int64_t> dims_mask{3, 2048};
  ge::Shape shape_mask(dims_mask);
  ge::TensorDesc tensorDescMask(shape_mask, FORMAT_ND, DT_UINT8);
  input_data_mask.update_input_desc_x(tensorDescMask);
  input_data_mask.update_output_desc_y(tensorDescMask);

  auto dropout_domask_op = op::DropOutDoMaskV3D("dropout_domask_v3d_0");
  dropout_domask_op.set_input_x(mid_cast_op, "y");
  dropout_domask_op.set_input_mask(input_data_mask);
  dropout_domask_op.set_attr_keep_prob(0.5f);

  auto end_op_pd_x = op::Cast("const0_op");
  end_op_pd_x.set_input_x(mid_cast_op, "y")
  .set_attr_dst_type(1);

  auto end_op_pd_x_dropout = op::Cast("const1_op");
  end_op_pd_x_dropout.set_input_x(dropout_domask_op, "y")
  .set_attr_dst_type(1);

  auto end_op_pd_gamma = op::Cast("const2_op");
  end_op_pd_gamma.set_input_x(layer_norm_grad_op, "pd_gamma")
  .set_attr_dst_type(1);

  auto end_op_pd_beta = op::Cast("const3_op");
  end_op_pd_beta.set_input_x(layer_norm_grad_op, "pd_beta")
  .set_attr_dst_type(1);

  std::vector<Operator> inputs{input_data, input_data_dy, input_data_mean, input_data_variance,
                               input_data_gamma, input_data_mask};
  std::vector<Operator> outputs{end_op_pd_x, end_op_pd_x_dropout, end_op_pd_gamma, end_op_pd_beta};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormGradDropOutDoMaskV3DFusionPass",
  fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
  std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
  fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
  fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradDropOutDoMaskV3DFusionPass"].GetMatchTimes(), 1);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradDropOutDoMaskV3DFusionPass"].GetEffectTimes(), 0);
}


TEST_F(ln_dropout_grad_fusion_pass_test, ln_dropout_grad_fusion_pass_test_6) {
  ge::Graph graph("ln_dropout_grad_fusion_pass_test_invalid_x_dims");
  auto input_data = op::Data("input_data");
  std::vector<int64_t> dims{1024};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_FLOAT16);
  input_data.update_input_desc_x(tensorDesc);
  input_data.update_output_desc_y(tensorDesc);

  auto input_data_dy = op::Data("input_data_dy");
  std::vector<int64_t> dims_dy{1024};
  ge::Shape shape_dy(dims_dy);
  ge::TensorDesc tensorDescDy(shape_dy, FORMAT_ND, DT_FLOAT16);
  input_data_dy.update_input_desc_x(tensorDescDy);
  input_data_dy.update_output_desc_y(tensorDescDy);

  auto input_data_mean = op::Data("input_data_mean");
  std::vector<int64_t> dims_mean{1};
  ge::Shape shape_mean(dims_mean);
  ge::TensorDesc tensorDescMean(shape_mean, FORMAT_ND, DT_FLOAT16);
  input_data_mean.update_input_desc_x(tensorDescMean);
  input_data_mean.update_output_desc_y(tensorDescMean);

  auto input_data_variance = op::Data("input_data_variance");
  std::vector<int64_t> dims_variance{1};
  ge::Shape shape_variance(dims_variance);
  ge::TensorDesc tensorDescVariance(shape_variance, FORMAT_ND, DT_FLOAT16);
  input_data_variance.update_input_desc_x(tensorDescVariance);
  input_data_variance.update_output_desc_y(tensorDescVariance);

  auto input_data_gamma = op::Data("input_data_gamma");
  std::vector<int64_t> dims_gamma{1024};
  ge::Shape shape_gamma(dims_gamma);
  ge::TensorDesc tensorDescGamma(shape_gamma, FORMAT_ND, DT_FLOAT16);
  input_data_gamma.update_input_desc_x(tensorDescGamma);
  input_data_gamma.update_output_desc_y(tensorDescGamma);

  auto layer_norm_grad_op = op::LayerNormGrad("layer_norm_grad_0");
  layer_norm_grad_op.set_input_x(input_data);
  layer_norm_grad_op.set_input_dy(input_data_dy);
  layer_norm_grad_op.set_input_mean(input_data_mean);
  layer_norm_grad_op.set_input_variance(input_data_variance);
  layer_norm_grad_op.set_input_gamma(input_data_gamma);

  auto mid_cast_op = op::Cast("mid_cast");
  mid_cast_op.set_input_x(layer_norm_grad_op, "pd_x")
  .set_attr_dst_type(0);

  auto input_data_mask = op::Data("input_mask");
  std::vector<int64_t> dims_mask{1024};
  ge::Shape shape_mask(dims_mask);
  ge::TensorDesc tensorDescMask(shape_mask, FORMAT_ND, DT_UINT8);
  input_data_mask.update_input_desc_x(tensorDescMask);
  input_data_mask.update_output_desc_y(tensorDescMask);

  auto dropout_domask_op = op::DropOutDoMaskV3D("dropout_domask_v3d_0");
  dropout_domask_op.set_input_x(mid_cast_op, "y");
  dropout_domask_op.set_input_mask(input_data_mask);
  dropout_domask_op.set_attr_keep_prob(0.5f);

  auto end_op_pd_x = op::Cast("const0_op");
  end_op_pd_x.set_input_x(mid_cast_op, "y")
  .set_attr_dst_type(1);

  auto end_op_pd_x_dropout = op::Cast("const1_op");
  end_op_pd_x_dropout.set_input_x(dropout_domask_op, "y")
  .set_attr_dst_type(1);

  auto end_op_pd_gamma = op::Cast("const2_op");
  end_op_pd_gamma.set_input_x(layer_norm_grad_op, "pd_gamma")
  .set_attr_dst_type(1);

  auto end_op_pd_beta = op::Cast("const3_op");
  end_op_pd_beta.set_input_x(layer_norm_grad_op, "pd_beta")
  .set_attr_dst_type(1);

  std::vector<Operator> inputs{input_data, input_data_dy, input_data_mean, input_data_variance,
                               input_data_gamma, input_data_mask};
  std::vector<Operator> outputs{end_op_pd_x, end_op_pd_x_dropout, end_op_pd_gamma, end_op_pd_beta};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormGradDropOutDoMaskV3DFusionPass",
  fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
  std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
  fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
  fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradDropOutDoMaskV3DFusionPass"].GetMatchTimes(), 1);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradDropOutDoMaskV3DFusionPass"].GetEffectTimes(), 0);
}

TEST_F(ln_dropout_grad_fusion_pass_test, ln_dropout_grad_fusion_pass_test_7) {
  ge::Graph graph("ln_dropout_grad_fusion_pass_test_unsupported_version");
  auto input_data = op::Data("input_data");
  std::vector<int64_t> dims{3, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_FLOAT16);
  input_data.update_input_desc_x(tensorDesc);
  input_data.update_output_desc_y(tensorDesc);

  auto input_data_dy = op::Data("input_data_dy");
  std::vector<int64_t> dims_dy{3, 32};
  ge::Shape shape_dy(dims_dy);
  ge::TensorDesc tensorDescDy(shape_dy, FORMAT_ND, DT_FLOAT16);
  input_data_dy.update_input_desc_x(tensorDescDy);
  input_data_dy.update_output_desc_y(tensorDescDy);

  auto input_data_mean = op::Data("input_data_mean");
  std::vector<int64_t> dims_mean{3, 1};
  ge::Shape shape_mean(dims_mean);
  ge::TensorDesc tensorDescMean(shape_mean, FORMAT_ND, DT_FLOAT16);
  input_data_mean.update_input_desc_x(tensorDescMean);
  input_data_mean.update_output_desc_y(tensorDescMean);

  auto input_data_variance = op::Data("input_data_variance");
  std::vector<int64_t> dims_variance{3, 1};
  ge::Shape shape_variance(dims_variance);
  ge::TensorDesc tensorDescVariance(shape_variance, FORMAT_ND, DT_FLOAT16);
  input_data_variance.update_input_desc_x(tensorDescVariance);
  input_data_variance.update_output_desc_y(tensorDescVariance);

  auto input_data_gamma = op::Data("input_data_gamma");
  std::vector<int64_t> dims_gamma{32};
  ge::Shape shape_gamma(dims_gamma);
  ge::TensorDesc tensorDescGamma(shape_gamma, FORMAT_ND, DT_FLOAT16);
  input_data_gamma.update_input_desc_x(tensorDescGamma);
  input_data_gamma.update_output_desc_y(tensorDescGamma);

  auto layer_norm_grad_op = op::LayerNormGrad("layer_norm_grad_0");
  layer_norm_grad_op.set_input_x(input_data);
  layer_norm_grad_op.set_input_dy(input_data_dy);
  layer_norm_grad_op.set_input_mean(input_data_mean);
  layer_norm_grad_op.set_input_variance(input_data_variance);
  layer_norm_grad_op.set_input_gamma(input_data_gamma);

  auto mid_cast_op = op::Cast("mid_cast");
  mid_cast_op.set_input_x(layer_norm_grad_op, "pd_x")
  .set_attr_dst_type(0);

  auto input_data_mask = op::Data("input_mask");
  std::vector<int64_t> dims_mask{3, 32};
  ge::Shape shape_mask(dims_mask);
  ge::TensorDesc tensorDescMask(shape_mask, FORMAT_ND, DT_UINT8);
  input_data_mask.update_input_desc_x(tensorDescMask);
  input_data_mask.update_output_desc_y(tensorDescMask);

  auto dropout_domask_op = op::DropOutDoMaskV3D("dropout_domask_v3d_0");
  dropout_domask_op.set_input_x(mid_cast_op, "y");
  dropout_domask_op.set_input_mask(input_data_mask);
  dropout_domask_op.set_attr_keep_prob(0.5f);

  auto end_op_pd_x = op::Cast("const0_op");
  end_op_pd_x.set_input_x(mid_cast_op, "y")
  .set_attr_dst_type(1);

  auto end_op_pd_x_dropout = op::Cast("const1_op");
  end_op_pd_x_dropout.set_input_x(dropout_domask_op, "y")
  .set_attr_dst_type(1);

  auto end_op_pd_gamma = op::Cast("const2_op");
  end_op_pd_gamma.set_input_x(layer_norm_grad_op, "pd_gamma")
  .set_attr_dst_type(1);

  auto end_op_pd_beta = op::Cast("const3_op");
  end_op_pd_beta.set_input_x(layer_norm_grad_op, "pd_beta")
  .set_attr_dst_type(1);

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 32;
  opti_compilation_info.soc_version = "Ascend710";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platform_info;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

  std::vector<Operator> inputs{input_data, input_data_dy, input_data_mean, input_data_variance,
                               input_data_gamma, input_data_mask};
  std::vector<Operator> outputs{end_op_pd_x, end_op_pd_x_dropout, end_op_pd_gamma, end_op_pd_beta};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormGradDropOutDoMaskV3DFusionPass",
  fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
  std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
  fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
  fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradDropOutDoMaskV3DFusionPass"].GetMatchTimes(), 1);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradDropOutDoMaskV3DFusionPass"].GetEffectTimes(), 0);
}
