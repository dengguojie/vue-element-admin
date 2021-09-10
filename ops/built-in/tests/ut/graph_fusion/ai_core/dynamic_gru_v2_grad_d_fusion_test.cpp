#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "rnn.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class dynamic_gru_v2_grad_d_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dynamic_gru_v2_grad_d_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dynamic_gru_v2_grad_d_fusion_test TearDown" << std::endl;
  }
};

TEST_F(dynamic_gru_v2_grad_d_fusion_test, dynamic_gru_v2_grad_d_fusion_test_1) {
  ge::Graph graph("dynamic_gru_v2_grad_d_fusion_test_1");
  int64_t tSize = -1;
  int64_t batchSize = 16;
  int64_t inputSize = 16;
  int64_t hiddenSize = 16;
  int64_t hiddenGateSize = 3 * hiddenSize;

  std::vector<int64_t> xVec{tSize, batchSize, inputSize};
  ge::Shape xShape(xVec);
  ge::TensorDesc xDesc(xShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> weightInputVec{inputSize, hiddenGateSize};
  ge::Shape weightInputShape(weightInputVec);
  ge::TensorDesc weightInputDesc(weightInputShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> weightHiddenVec{hiddenSize, hiddenGateSize};
  ge::Shape weightHiddenShape(weightHiddenVec);
  ge::TensorDesc weightHiddenDesc(weightHiddenShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> noTVec{batchSize, hiddenSize};
  ge::Shape noTShape(noTVec);
  std::vector<int64_t> tVec{tSize, batchSize, hiddenSize};
  ge::Shape tShape(tVec);
  ge::TensorDesc yDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc initHDesc(noTShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc hDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc dyDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc dhDesc(noTShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc updateDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc resetDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc newDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc hiddenNewDesc(tShape, FORMAT_ND, DT_FLOAT16);

  auto xData = op::Data("x");
  xData.update_input_desc_x(xDesc);
  xData.update_output_desc_y(xDesc);
  auto weightInputData = op::Data("weight_input");
  weightInputData.update_input_desc_x(weightInputDesc);
  weightInputData.update_output_desc_y(weightInputDesc);
  auto weightHiddenData = op::Data("weight_hidden");
  weightHiddenData.update_input_desc_x(weightHiddenDesc);
  weightHiddenData.update_output_desc_y(weightHiddenDesc);
  auto yData = op::Data("y");
  yData.update_input_desc_x(yDesc);
  yData.update_output_desc_y(yDesc);
  auto initHData = op::Data("init_h");
  initHData.update_input_desc_x(initHDesc);
  initHData.update_output_desc_y(initHDesc);
  auto hData = op::Data("h");
  hData.update_input_desc_x(hDesc);
  hData.update_output_desc_y(hDesc);
  auto dyData = op::Data("dy");
  dyData.update_input_desc_x(dyDesc);
  dyData.update_output_desc_y(dyDesc);
  auto dhData = op::Data("dh");
  dhData.update_input_desc_x(dhDesc);
  dhData.update_output_desc_y(dhDesc);
  auto updateData = op::Data("update");
  updateData.update_input_desc_x(updateDesc);
  updateData.update_output_desc_y(updateDesc);
  auto resetData = op::Data("reset");
  resetData.update_input_desc_x(resetDesc);
  resetData.update_output_desc_y(resetDesc);
  auto newData = op::Data("new");
  newData.update_input_desc_x(newDesc);
  newData.update_output_desc_y(newDesc);
  auto hiddenNewData = op::Data("hidden_new");
  hiddenNewData.update_input_desc_x(hiddenNewDesc);
  hiddenNewData.update_output_desc_y(hiddenNewDesc);

  auto dynamicGruV2GradOp = op::DynamicGRUV2Grad("DynamicGRUV2Grad_1");
  dynamicGruV2GradOp.set_input_x(xData)
      .set_input_weight_input(weightInputData)
      .set_input_weight_hidden(weightHiddenData)
      .set_input_y(yData)
      .set_input_init_h(initHData)
      .set_input_h(hData)
      .set_input_dy(dyData)
      .set_input_dh(dhData)
      .set_input_update(updateData)
      .set_input_reset(resetData)
      .set_input_new(newData)
      .set_input_hidden_new(hiddenNewData);

  std::vector<Operator> inputs{xData, weightInputData, weightHiddenData};
  std::vector<Operator> outputs{dynamicGruV2GradOp};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraphPtr = ge::GraphUtils::GetComputeGraph(graph);
  std::string session_id = "testGraph";
  auto res = AttrUtils::SetStr(computeGraphPtr, "_session_graph_id", session_id);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraphPtr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicGRUV2GradDFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraphPtr);

  bool findDynamicGRUCellGrad = false;
  for (auto node : computeGraphPtr->GetAllNodes()) {
    if (node->GetType() == "DynamicGRUCellGrad") {
      findDynamicGRUCellGrad = true;
    }
  }
  EXPECT_EQ(findDynamicGRUCellGrad, true);
}

TEST_F(dynamic_gru_v2_grad_d_fusion_test, dynamic_gru_v2_grad_d_fusion_test_2) {
  ge::Graph graph("dynamic_gru_v2_grad_d_fusion_test_2");
  int64_t tSize = -1;
  int64_t batchSize = 16;
  int64_t inputSize = 768;
  int64_t hiddenSize = 512;
  int64_t hiddenGateSize = 3 * hiddenSize;

  std::vector<int64_t> xVec{tSize, batchSize, inputSize};
  ge::Shape xShape(xVec);
  ge::TensorDesc xDesc(xShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> weightInputVec{inputSize, hiddenGateSize};
  ge::Shape weightInputShape(weightInputVec);
  ge::TensorDesc weightInputDesc(weightInputShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> weightHiddenVec{hiddenSize, hiddenGateSize};
  ge::Shape weightHiddenShape(weightHiddenVec);
  ge::TensorDesc weightHiddenDesc(weightHiddenShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> noTVec{batchSize, hiddenSize};
  ge::Shape noTShape(noTVec);
  std::vector<int64_t> tVec{tSize, batchSize, hiddenSize};
  ge::Shape tShape(tVec);
  ge::TensorDesc yDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc initHDesc(noTShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc hDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc dyDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc dhDesc(noTShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc updateDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc resetDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc newDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc hiddenNewDesc(tShape, FORMAT_ND, DT_FLOAT16);

  auto xData = op::Data("x");
  xData.update_input_desc_x(xDesc);
  xData.update_output_desc_y(xDesc);
  auto weightInputData = op::Data("weight_input");
  weightInputData.update_input_desc_x(weightInputDesc);
  weightInputData.update_output_desc_y(weightInputDesc);
  auto weightHiddenData = op::Data("weight_hidden");
  weightHiddenData.update_input_desc_x(weightHiddenDesc);
  weightHiddenData.update_output_desc_y(weightHiddenDesc);
  auto yData = op::Data("y");
  yData.update_input_desc_x(yDesc);
  yData.update_output_desc_y(yDesc);
  auto initHData = op::Data("init_h");
  initHData.update_input_desc_x(initHDesc);
  initHData.update_output_desc_y(initHDesc);
  auto hData = op::Data("h");
  hData.update_input_desc_x(hDesc);
  hData.update_output_desc_y(hDesc);
  auto dyData = op::Data("dy");
  dyData.update_input_desc_x(dyDesc);
  dyData.update_output_desc_y(dyDesc);
  auto dhData = op::Data("dh");
  dhData.update_input_desc_x(dhDesc);
  dhData.update_output_desc_y(dhDesc);
  auto updateData = op::Data("update");
  updateData.update_input_desc_x(updateDesc);
  updateData.update_output_desc_y(updateDesc);
  auto resetData = op::Data("reset");
  resetData.update_input_desc_x(resetDesc);
  resetData.update_output_desc_y(resetDesc);
  auto newData = op::Data("new");
  newData.update_input_desc_x(newDesc);
  newData.update_output_desc_y(newDesc);
  auto hiddenNewData = op::Data("hidden_new");
  hiddenNewData.update_input_desc_x(hiddenNewDesc);
  hiddenNewData.update_output_desc_y(hiddenNewDesc);

  auto dynamicGruV2GradOp = op::DynamicGRUV2Grad("DynamicGRUV2Grad_1");
  dynamicGruV2GradOp.set_input_x(xData)
      .set_input_weight_input(weightInputData)
      .set_input_weight_hidden(weightHiddenData)
      .set_input_y(yData)
      .set_input_init_h(initHData)
      .set_input_h(hData)
      .set_input_dy(dyData)
      .set_input_dh(dhData)
      .set_input_update(updateData)
      .set_input_reset(resetData)
      .set_input_new(newData)
      .set_input_hidden_new(hiddenNewData);

  std::vector<Operator> inputs{xData, weightInputData, weightHiddenData};
  std::vector<Operator> outputs{dynamicGruV2GradOp};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraphPtr = ge::GraphUtils::GetComputeGraph(graph);
  std::string session_id = "testGraph";
  auto res = AttrUtils::SetStr(computeGraphPtr, "_session_graph_id", session_id);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraphPtr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicGRUV2GradDFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraphPtr);

  bool findDynamicGRUCellGrad = false;
  for (auto node : computeGraphPtr->GetAllNodes()) {
    if (node->GetType() == "DynamicGRUCellGrad") {
      findDynamicGRUCellGrad = true;
    }
  }
  EXPECT_EQ(findDynamicGRUCellGrad, true);
}

TEST_F(dynamic_gru_v2_grad_d_fusion_test, dynamic_gru_v2_grad_d_fusion_test_3) {
  ge::Graph graph("dynamic_gru_v2_grad_d_fusion_test_3");
  int64_t tSize = 1;
  int64_t batchSize = 16;
  int64_t inputSize = 768;
  int64_t hiddenSize = 512;
  int64_t hiddenGateSize = 3 * hiddenSize;

  std::vector<int64_t> xVec{tSize, batchSize, inputSize};
  ge::Shape xShape(xVec);
  ge::TensorDesc xDesc(xShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> weightInputVec{inputSize, hiddenGateSize};
  ge::Shape weightInputShape(weightInputVec);
  ge::TensorDesc weightInputDesc(weightInputShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> weightHiddenVec{hiddenSize, hiddenGateSize};
  ge::Shape weightHiddenShape(weightHiddenVec);
  ge::TensorDesc weightHiddenDesc(weightHiddenShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> noTVec{batchSize, hiddenSize};
  ge::Shape noTShape(noTVec);
  std::vector<int64_t> tVec{tSize, batchSize, hiddenSize};
  ge::Shape tShape(tVec);
  ge::TensorDesc yDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc initHDesc(noTShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc hDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc dyDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc dhDesc(noTShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc updateDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc resetDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc newDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc hiddenNewDesc(tShape, FORMAT_ND, DT_FLOAT16);

  auto xData = op::Data("x");
  xData.update_input_desc_x(xDesc);
  xData.update_output_desc_y(xDesc);
  auto weightInputData = op::Data("weight_input");
  weightInputData.update_input_desc_x(weightInputDesc);
  weightInputData.update_output_desc_y(weightInputDesc);
  auto weightHiddenData = op::Data("weight_hidden");
  weightHiddenData.update_input_desc_x(weightHiddenDesc);
  weightHiddenData.update_output_desc_y(weightHiddenDesc);
  auto yData = op::Data("y");
  yData.update_input_desc_x(yDesc);
  yData.update_output_desc_y(yDesc);
  auto initHData = op::Data("init_h");
  initHData.update_input_desc_x(initHDesc);
  initHData.update_output_desc_y(initHDesc);
  auto hData = op::Data("h");
  hData.update_input_desc_x(hDesc);
  hData.update_output_desc_y(hDesc);
  auto dyData = op::Data("dy");
  dyData.update_input_desc_x(dyDesc);
  dyData.update_output_desc_y(dyDesc);
  auto dhData = op::Data("dh");
  dhData.update_input_desc_x(dhDesc);
  dhData.update_output_desc_y(dhDesc);
  auto updateData = op::Data("update");
  updateData.update_input_desc_x(updateDesc);
  updateData.update_output_desc_y(updateDesc);
  auto resetData = op::Data("reset");
  resetData.update_input_desc_x(resetDesc);
  resetData.update_output_desc_y(resetDesc);
  auto newData = op::Data("new");
  newData.update_input_desc_x(newDesc);
  newData.update_output_desc_y(newDesc);
  auto hiddenNewData = op::Data("hidden_new");
  hiddenNewData.update_input_desc_x(hiddenNewDesc);
  hiddenNewData.update_output_desc_y(hiddenNewDesc);

  auto dynamicGruV2GradOp = op::DynamicGRUV2Grad("DynamicGRUV2Grad_1");
  dynamicGruV2GradOp.set_input_x(xData)
      .set_input_weight_input(weightInputData)
      .set_input_weight_hidden(weightHiddenData)
      .set_input_y(yData)
      .set_input_init_h(initHData)
      .set_input_h(hData)
      .set_input_dy(dyData)
      .set_input_dh(dhData)
      .set_input_update(updateData)
      .set_input_reset(resetData)
      .set_input_new(newData)
      .set_input_hidden_new(hiddenNewData);

  std::vector<Operator> inputs{xData, weightInputData, weightHiddenData};
  std::vector<Operator> outputs{dynamicGruV2GradOp};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraphPtr = ge::GraphUtils::GetComputeGraph(graph);
  std::string session_id = "testGraph";
  auto res = AttrUtils::SetStr(computeGraphPtr, "_session_graph_id", session_id);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraphPtr);
  fe::Status status = fe::FusionPassTestUtils::RunGraphFusionPass("DynamicGRUV2GradDFusionPass",
                                                                  fe::BUILT_IN_GRAPH_PASS, *computeGraphPtr);

  EXPECT_EQ(status, fe::NOT_CHANGED);
}

TEST_F(dynamic_gru_v2_grad_d_fusion_test, dynamic_gru_v2_grad_d_fusion_test_4) {
  ge::Graph graph("dynamic_gru_v2_grad_d_fusion_test_4");
  int64_t tSize = 1;
  int64_t batchSize = 16;
  int64_t inputSize = -1;
  int64_t hiddenSize = 512;
  int64_t hiddenGateSize = 3 * hiddenSize;

  std::vector<int64_t> xVec{tSize, batchSize, inputSize};
  ge::Shape xShape(xVec);
  ge::TensorDesc xDesc(xShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> weightInputVec{inputSize, hiddenGateSize};
  ge::Shape weightInputShape(weightInputVec);
  ge::TensorDesc weightInputDesc(weightInputShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> weightHiddenVec{hiddenSize, hiddenGateSize};
  ge::Shape weightHiddenShape(weightHiddenVec);
  ge::TensorDesc weightHiddenDesc(weightHiddenShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> noTVec{batchSize, hiddenSize};
  ge::Shape noTShape(noTVec);
  std::vector<int64_t> tVec{tSize, batchSize, hiddenSize};
  ge::Shape tShape(tVec);
  ge::TensorDesc yDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc initHDesc(noTShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc hDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc dyDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc dhDesc(noTShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc updateDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc resetDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc newDesc(tShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc hiddenNewDesc(tShape, FORMAT_ND, DT_FLOAT16);

  auto xData = op::Data("x");
  xData.update_input_desc_x(xDesc);
  xData.update_output_desc_y(xDesc);
  auto weightInputData = op::Data("weight_input");
  weightInputData.update_input_desc_x(weightInputDesc);
  weightInputData.update_output_desc_y(weightInputDesc);
  auto weightHiddenData = op::Data("weight_hidden");
  weightHiddenData.update_input_desc_x(weightHiddenDesc);
  weightHiddenData.update_output_desc_y(weightHiddenDesc);
  auto yData = op::Data("y");
  yData.update_input_desc_x(yDesc);
  yData.update_output_desc_y(yDesc);
  auto initHData = op::Data("init_h");
  initHData.update_input_desc_x(initHDesc);
  initHData.update_output_desc_y(initHDesc);
  auto hData = op::Data("h");
  hData.update_input_desc_x(hDesc);
  hData.update_output_desc_y(hDesc);
  auto dyData = op::Data("dy");
  dyData.update_input_desc_x(dyDesc);
  dyData.update_output_desc_y(dyDesc);
  auto dhData = op::Data("dh");
  dhData.update_input_desc_x(dhDesc);
  dhData.update_output_desc_y(dhDesc);
  auto updateData = op::Data("update");
  updateData.update_input_desc_x(updateDesc);
  updateData.update_output_desc_y(updateDesc);
  auto resetData = op::Data("reset");
  resetData.update_input_desc_x(resetDesc);
  resetData.update_output_desc_y(resetDesc);
  auto newData = op::Data("new");
  newData.update_input_desc_x(newDesc);
  newData.update_output_desc_y(newDesc);
  auto hiddenNewData = op::Data("hidden_new");
  hiddenNewData.update_input_desc_x(hiddenNewDesc);
  hiddenNewData.update_output_desc_y(hiddenNewDesc);

  auto dynamicGruV2GradOp = op::DynamicGRUV2Grad("DynamicGRUV2Grad_1");
  dynamicGruV2GradOp.set_input_x(xData)
      .set_input_weight_input(weightInputData)
      .set_input_weight_hidden(weightHiddenData)
      .set_input_y(yData)
      .set_input_init_h(initHData)
      .set_input_h(hData)
      .set_input_dy(dyData)
      .set_input_dh(dhData)
      .set_input_update(updateData)
      .set_input_reset(resetData)
      .set_input_new(newData)
      .set_input_hidden_new(hiddenNewData);

  std::vector<Operator> inputs{xData, weightInputData, weightHiddenData};
  std::vector<Operator> outputs{dynamicGruV2GradOp};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraphPtr = ge::GraphUtils::GetComputeGraph(graph);
  std::string session_id = "testGraph";
  auto res = AttrUtils::SetStr(computeGraphPtr, "_session_graph_id", session_id);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraphPtr);
  fe::Status status = fe::FusionPassTestUtils::RunGraphFusionPass("DynamicGRUV2GradDFusionPass",
                                                                  fe::BUILT_IN_GRAPH_PASS, *computeGraphPtr);

  EXPECT_EQ(status, fe::NOT_CHANGED);
}