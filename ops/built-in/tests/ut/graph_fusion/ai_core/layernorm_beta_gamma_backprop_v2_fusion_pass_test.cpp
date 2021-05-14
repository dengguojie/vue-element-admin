//
// Created by c30002892 on 2020/9/5.
//

#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_norm_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
using namespace op;

class layernorm_beta_gamma_backprop_v2_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "inplace_add TearDown" << std::endl;
    }

};

TEST_F(layernorm_beta_gamma_backprop_v2_fusion_pass_test, layernorm_beta_gamma_backprop_v2_fusion_pass_test_1) {

  ge::Graph graph("layernorm_beta_gamma_backprop_v2_fusion_pass_test_1");
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

  auto layer_norm_beta_gamma_backprop_op = op::LayerNormBetaGammaBackpropV2("layer_norm_beta_0");
  layer_norm_beta_gamma_backprop_op.set_input_res_for_gamma(input_data);
  layer_norm_beta_gamma_backprop_op.set_input_dy(input_data_dy);

  auto end_op_pd_gamma = op::Cast("const0_op");
  end_op_pd_gamma.set_input_x(layer_norm_beta_gamma_backprop_op, "pd_gamma")
                 .set_attr_dst_type(0);

  auto end_op_pd_beta = op::Cast("const1_op");
  end_op_pd_beta.set_input_x(layer_norm_beta_gamma_backprop_op, "pd_beta")
                .set_attr_dst_type(1);

  std::vector<Operator> inputs{input_data_dy, input_data};
  std::vector<Operator> outputs{end_op_pd_gamma, end_op_pd_beta};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormGradFusionPassBetaGammaV2", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
  std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
  fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
  fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradFusionPassBetaGammaV2"].GetMatchTimes(), 1);
  EXPECT_EQ(graphFusionInfoMap["LayerNormGradFusionPassBetaGammaV2"].GetEffectTimes(), 1);
}
