#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "rnn.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "fp16_t.hpp"

using namespace ge;
using namespace op;

class dynamic_augru_seqlength_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", true);
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", true);
    std::cout << "dynamic_augru_seqlength_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dynamic_augru_seqlength_fusion_test TearDown" << std::endl;
  }
};

TEST_F(dynamic_augru_seqlength_fusion_test, dynamic_augru_seqlength_fusion_test_1) {
  ge::Graph graph("dynamic_augru_seqlength_fusion_test_1");

  int64_t tSize = 1;
  int64_t batchSize = 128;
  int64_t inputSize = 64;
  int64_t hiddenSize = 32;
  int64_t hiddenGateSize = 3 * hiddenSize;

  std::vector<int64_t> xVec{tSize, batchSize, inputSize};
  ge::Shape xShape(xVec);
  ge::TensorDesc xDesc(xShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> weightInputVec{inputSize, hiddenGateSize};
  ge::Shape weightInputShape(weightInputVec);
  ge::TensorDesc weightInputDesc(weightInputShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> weightHiddenVec{tSize, hiddenSize, hiddenGateSize};
  ge::Shape weightHiddenShape(weightHiddenVec);
  ge::TensorDesc weightHiddenDesc(weightHiddenShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> weightAttVec{tSize, batchSize};
  ge::Shape weightAttShape(weightAttVec);
  ge::TensorDesc weightAttDesc(weightAttShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> biasVec{hiddenGateSize};
  ge::Shape biasShape(biasVec);
  ge::TensorDesc biasinputDesc(biasShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc biashiddenDesc(biasShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> seqLenVec{batchSize};
  ge::Shape seqLenShape(seqLenVec);
  ge::TensorDesc seqLenDesc(seqLenShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> initHVec{batchSize, hiddenSize};
  ge::Shape initHShape(initHVec);
  ge::TensorDesc initHDesc(initHShape, FORMAT_ND, DT_FLOAT16);

  auto xData = op::Data("x");
  xData.update_input_desc_x(xDesc);
  xData.update_output_desc_y(xDesc);

  auto weightInputData = op::Data("weight_input");
  weightInputData.update_input_desc_x(weightInputDesc);
  weightInputData.update_output_desc_y(weightInputDesc);

  auto weightHiddenData = op::Data("weight_hidden");
  weightHiddenData.update_input_desc_x(weightHiddenDesc);
  weightHiddenData.update_output_desc_y(weightHiddenDesc);

  auto weightAttData = op::Data("weight_att");
  weightAttData.update_input_desc_x(weightAttDesc);
  weightAttData.update_output_desc_y(weightAttDesc);

  auto biasInputData = op::Data("bias_input");
  biasInputData.update_input_desc_x(biasinputDesc);
  biasInputData.update_output_desc_y(biasinputDesc);

  auto biasHiddenData = op::Data("bias_hidden");
  biasHiddenData.update_input_desc_x(biashiddenDesc);
  biasHiddenData.update_output_desc_y(biashiddenDesc);

  auto seqLengthData = op::Data("seq_length");
  seqLengthData.update_input_desc_x(seqLenDesc);
  seqLengthData.update_output_desc_y(seqLenDesc);

  auto initHData = op::Data("init_h");
  initHData.update_input_desc_x(initHDesc);
  initHData.update_output_desc_y(initHDesc);

  auto rnn_op = op::DynamicAUGRU("DynamicAUGRU");
  rnn_op.set_input_x(xData)
      .set_input_weight_input(weightInputData)
      .set_input_weight_hidden(weightHiddenData)
      .set_input_weight_att(weightAttData)
      .set_input_bias_input(biasInputData)
      .set_input_bias_hidden(biasHiddenData)
      .set_input_seq_length(seqLengthData)
      .set_input_init_h(initHData);
  std::vector<Operator> inputs{xData,         weightInputData, weightHiddenData, weightAttData,
                               biasInputData, biasHiddenData,  seqLengthData,    initHData};
  std::vector<Operator> outputs{rnn_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicAUGRUAddSeqPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool find_gen_mask = false;
  bool find_rnn = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "RnnGenMask" || node->GetType() == "RnnGenMaskV2") {
      find_gen_mask = true;
    }

    if (node->GetType() == "DynamicAUGRU") {
      find_rnn = true;
    }
  }
  EXPECT_EQ(find_gen_mask, true);
}

TEST_F(dynamic_augru_seqlength_fusion_test, dynamic_augru_seqlength_fusion_test_2) {
  ge::Graph graph("dynamic_augru_seqlength_fusion_test_2");

  int64_t tSize = 1;
  int64_t batchSize = 128;
  int64_t inputSize = 64;
  int64_t hiddenSize = 30;
  int64_t hiddenGateSize = 3 * hiddenSize;

  std::vector<int64_t> xVec{-1, batchSize, -1};
  ge::Shape xShape(xVec);
  ge::TensorDesc xDesc(xShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> weightInputVec{inputSize, hiddenGateSize};
  ge::Shape weightInputShape(weightInputVec);
  ge::TensorDesc weightInputDesc(weightInputShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> weightHiddenVec{tSize, hiddenSize, hiddenGateSize};
  ge::Shape weightHiddenShape(weightHiddenVec);
  ge::TensorDesc weightHiddenDesc(weightHiddenShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> weightAttVec{tSize, batchSize};
  ge::Shape weightAttShape(weightAttVec);
  ge::TensorDesc weightAttDesc(weightAttShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> biasVec{hiddenGateSize};
  ge::Shape biasShape(biasVec);
  ge::TensorDesc biasinputDesc(biasShape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc biashiddenDesc(biasShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> seqLenVec{batchSize};
  ge::Shape seqLenShape(seqLenVec);
  ge::TensorDesc seqLenDesc(seqLenShape, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> initHVec{batchSize, hiddenSize};
  ge::Shape initHShape(initHVec);
  ge::TensorDesc initHDesc(initHShape, FORMAT_ND, DT_FLOAT16);

  auto xData = op::Data("x");
  xData.update_input_desc_x(xDesc);
  xData.update_output_desc_y(xDesc);

  auto weightInputData = op::Data("weight_input");
  weightInputData.update_input_desc_x(weightInputDesc);
  weightInputData.update_output_desc_y(weightInputDesc);

  auto weightHiddenData = op::Data("weight_hidden");
  weightHiddenData.update_input_desc_x(weightHiddenDesc);
  weightHiddenData.update_output_desc_y(weightHiddenDesc);

  auto weightAttData = op::Data("weight_att");
  weightAttData.update_input_desc_x(weightAttDesc);
  weightAttData.update_output_desc_y(weightAttDesc);

  auto biasInputData = op::Data("bias_input");
  biasInputData.update_input_desc_x(biasinputDesc);
  biasInputData.update_output_desc_y(biasinputDesc);

  auto biasHiddenData = op::Data("bias_hidden");
  biasHiddenData.update_input_desc_x(biashiddenDesc);
  biasHiddenData.update_output_desc_y(biashiddenDesc);

  auto seqLengthData = op::Data("seq_length");
  seqLengthData.update_input_desc_x(seqLenDesc);
  seqLengthData.update_output_desc_y(seqLenDesc);

  auto initHData = op::Data("init_h");
  initHData.update_input_desc_x(initHDesc);
  initHData.update_output_desc_y(initHDesc);

  auto rnn_op = op::DynamicAUGRU("DynamicAUGRU");
  rnn_op.set_input_x(xData)
      .set_input_weight_input(weightInputData)
      .set_input_weight_hidden(weightHiddenData)
      .set_input_weight_att(weightAttData)
      .set_input_bias_input(biasInputData)
      .set_input_bias_hidden(biasHiddenData)
      .set_input_seq_length(seqLengthData)
      .set_input_init_h(initHData);
  std::vector<Operator> inputs{xData,         weightInputData, weightHiddenData, weightAttData,
                               biasInputData, biasHiddenData,  seqLengthData,    initHData};
  std::vector<Operator> outputs{rnn_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicAUGRUAddSeqPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool find_gen_mask = false;
  bool find_rnn = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "RnnGenMask" || node->GetType() == "RnnGenMaskV2") {
      find_gen_mask = true;
    }

    if (node->GetType() == "DynamicAUGRU") {
      find_rnn = true;
    }
  }
  EXPECT_EQ(find_gen_mask, true);
}