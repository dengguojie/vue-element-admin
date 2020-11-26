//
// Created by c30002892 on 2020/9/5.
//

#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_training_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
using namespace op;

class apply_addoutput_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "inplace_add TearDown" << std::endl;
    }
};

TEST_F(apply_addoutput_fusion_test, apply_addoutput_fusion_test_1) {
  ge::Graph graph("apply_addoutput_fusion_test_1");

  auto input_data_var = op::Data("input_data_var");
  std::vector<int64_t> dims_var{3, 32};
  ge::Shape shape_var(dims_var);
  ge::TensorDesc tensorDescVar(shape_var);
  input_data_var.update_input_desc_x(tensorDescVar);
  input_data_var.update_output_desc_y(tensorDescVar);

  auto input_data_ms = op::Data("input_data_ms");
  std::vector<int64_t> dims_ms{3, 32};
  ge::Shape shape_ms(dims_ms);
  ge::TensorDesc tensorDescMs(shape_ms);
  input_data_ms.update_input_desc_x(tensorDescMs);
  input_data_ms.update_output_desc_y(tensorDescMs);

  auto input_data_mom = op::Data("input_data_mom");
  std::vector<int64_t> dims_mom{3, 32};
  ge::Shape shape_mom(dims_mom);
  ge::TensorDesc tensorDescMom(shape_mom);
  input_data_mom.update_input_desc_x(tensorDescMom);
  input_data_mom.update_output_desc_y(tensorDescMom);

  auto input_data_lr = op::Data("input_data_lr");
  std::vector<int64_t> dims_lr{3, 32};
  ge::Shape shape_lr(dims_lr);
  ge::TensorDesc tensorDescLr(shape_lr);
  input_data_lr.update_input_desc_x(tensorDescLr);
  input_data_lr.update_output_desc_y(tensorDescLr);

  auto input_data_rho = op::Data("input_data_rho");
  std::vector<int64_t> dims_rho{3, 32};
  ge::Shape shape_rho(dims_rho);
  ge::TensorDesc tensorDescRho(shape_rho);
  input_data_rho.update_input_desc_x(tensorDescRho);
  input_data_rho.update_output_desc_y(tensorDescRho);

  auto input_data_momentum = op::Data("input_data_momentum");
  std::vector<int64_t> dims_momentum{3, 32};
  ge::Shape shape_momentum(dims_momentum);
  ge::TensorDesc tensorDescMomentum(shape_momentum);
  input_data_momentum.update_input_desc_x(tensorDescMomentum);
  input_data_momentum.update_output_desc_y(tensorDescMomentum);

  auto input_data_epsilon = op::Data("input_data_epsilon");
  std::vector<int64_t> dims_epsilon{3, 32};
  ge::Shape shape_epsilon(dims_epsilon);
  ge::TensorDesc tensorEpsilon(shape_epsilon);
  input_data_epsilon.update_input_desc_x(tensorEpsilon);
  input_data_epsilon.update_output_desc_y(tensorEpsilon);

  auto input_data_grad = op::Data("input_data_grad");
  std::vector<int64_t> dims_grad{3, 32};
  ge::Shape shape_grad(dims_var);
  ge::TensorDesc tensorGrad(shape_var);
  input_data_grad.update_input_desc_x(tensorGrad);
  input_data_grad.update_output_desc_y(tensorGrad);

  auto apply_rms_prop_op = op::ApplyRMSProp("apply_rms_prop_0");
  apply_rms_prop_op.set_input_var(input_data_var);
  apply_rms_prop_op.set_input_ms(input_data_ms);
  apply_rms_prop_op.set_input_mom(input_data_mom);
  apply_rms_prop_op.set_input_lr(input_data_lr);
  apply_rms_prop_op.set_input_rho(input_data_rho);
  apply_rms_prop_op.set_input_momentum(input_data_momentum);
  apply_rms_prop_op.set_input_epsilon(input_data_epsilon);
  apply_rms_prop_op.set_input_grad(input_data_grad);
  apply_rms_prop_op.set_attr_use_locking(false);

  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(apply_rms_prop_op);
  std::vector<Operator> inputs{input_data_var, input_data_ms, input_data_mom, input_data_lr,
                               input_data_rho, input_data_momentum};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "apply_addoutput_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ApplyAddOutputPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "apply_addoutput_fusion_test_1_after");
  std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
  std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
  fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
  fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
  EXPECT_EQ(graphFusionInfoMap["ApplyAddOutputPass"].GetMatchTimes(), 1);
  EXPECT_EQ(graphFusionInfoMap["ApplyAddOutputPass"].GetEffectTimes(), 1);
}

TEST_F(apply_addoutput_fusion_test, apply_addoutput_fusion_test_2) {
  ge::Graph graph("apply_addoutput_fusion_test_2");

  auto input_data_var = op::Data("input_data_var");
  std::vector<int64_t> dims_var{3, 32};
  ge::Shape shape_var(dims_var);
  ge::TensorDesc tensorDescVar(shape_var, FORMAT_ND, DT_INT8);
  input_data_var.update_input_desc_x(tensorDescVar);
  input_data_var.update_output_desc_y(tensorDescVar);

  auto input_data_ms = op::Data("input_data_ms");
  std::vector<int64_t> dims_ms{3, 32};
  ge::Shape shape_ms(dims_ms);
  ge::TensorDesc tensorDescMs(shape_ms);
  input_data_ms.update_input_desc_x(tensorDescMs);
  input_data_ms.update_output_desc_y(tensorDescMs);

  auto input_data_mom = op::Data("input_data_mom");
  std::vector<int64_t> dims_mom{3, 32};
  ge::Shape shape_mom(dims_mom);
  ge::TensorDesc tensorDescMom(shape_mom);
  input_data_mom.update_input_desc_x(tensorDescMom);
  input_data_mom.update_output_desc_y(tensorDescMom);

  auto input_data_lr = op::Data("input_data_lr");
  std::vector<int64_t> dims_lr{3, 32};
  ge::Shape shape_lr(dims_lr);
  ge::TensorDesc tensorDescLr(shape_lr);
  input_data_lr.update_input_desc_x(tensorDescLr);
  input_data_lr.update_output_desc_y(tensorDescLr);

  auto input_data_rho = op::Data("input_data_rho");
  std::vector<int64_t> dims_rho{3, 32};
  ge::Shape shape_rho(dims_rho);
  ge::TensorDesc tensorDescRho(shape_rho);
  input_data_rho.update_input_desc_x(tensorDescRho);
  input_data_rho.update_output_desc_y(tensorDescRho);

  auto input_data_momentum = op::Data("input_data_momentum");
  std::vector<int64_t> dims_momentum{3, 32};
  ge::Shape shape_momentum(dims_momentum);
  ge::TensorDesc tensorDescMomentum(shape_momentum);
  input_data_momentum.update_input_desc_x(tensorDescMomentum);
  input_data_momentum.update_output_desc_y(tensorDescMomentum);

  auto input_data_epsilon = op::Data("input_data_epsilon");
  std::vector<int64_t> dims_epsilon{3, 32};
  ge::Shape shape_epsilon(dims_epsilon);
  ge::TensorDesc tensorEpsilon(shape_epsilon);
  input_data_epsilon.update_input_desc_x(tensorEpsilon);
  input_data_epsilon.update_output_desc_y(tensorEpsilon);

  auto input_data_grad = op::Data("input_data_grad");
  std::vector<int64_t> dims_grad{3, 32};
  ge::Shape shape_grad(dims_var);
  ge::TensorDesc tensorGrad(shape_var);
  input_data_grad.update_input_desc_x(tensorGrad);
  input_data_grad.update_output_desc_y(tensorGrad);

  auto apply_rms_prop_op = op::ApplyRMSProp("apply_rms_prop_0");
  apply_rms_prop_op.set_input_var(input_data_var);
  apply_rms_prop_op.set_input_ms(input_data_ms);
  apply_rms_prop_op.set_input_mom(input_data_mom);
  apply_rms_prop_op.set_input_lr(input_data_lr);
  apply_rms_prop_op.set_input_rho(input_data_rho);
  apply_rms_prop_op.set_input_momentum(input_data_momentum);
  apply_rms_prop_op.set_input_epsilon(input_data_epsilon);
  apply_rms_prop_op.set_input_grad(input_data_grad);
  apply_rms_prop_op.set_attr_use_locking(false);

  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(apply_rms_prop_op);
  std::vector<Operator> inputs{input_data_var};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "apply_addoutput_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ApplyAddOutputPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "apply_addoutput_fusion_test_1_after");
  std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
  std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
  fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
  fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
  EXPECT_EQ(graphFusionInfoMap["ApplyAddOutputPass"].GetMatchTimes(), 1);
  EXPECT_EQ(graphFusionInfoMap["ApplyAddOutputPass"].GetEffectTimes(), 0);
}
