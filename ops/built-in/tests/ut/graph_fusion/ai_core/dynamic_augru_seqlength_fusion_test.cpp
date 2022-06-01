#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "rnn.h"
#include "array_ops.h"
#include "nn_norm_ops.h"
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

  std::vector<int64_t> weightAttVec{-1, batchSize};
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

#define DESC_DATA(name, shape_in, format_in, shape_out, format_out, dtype) \
  ge::GeTensorDesc desc_##name(shape_out, format_out, dtype);              \
  desc_##name.SetOriginFormat(format_in);                                  \
  desc_##name.SetOriginShape(shape_in)

TEST_F(dynamic_augru_seqlength_fusion_test, dynamic_augru_seqlength_fusion_test_3) {
  ge::Graph graph("dynamic_augru_seqlength_fusion_test_3");

  int64_t tSize = 1;
  int64_t batchSize = 128;
  int64_t inputSize = 64;
  int64_t hiddenSize = 32;
  int64_t hiddenGateSize = 3 * hiddenSize;

  DESC_DATA(xVec, ge::GeShape({tSize, batchSize, inputSize}), FORMAT_ND, ge::GeShape({tSize, batchSize, inputSize}),
            FORMAT_ND, DT_FLOAT16);
  DESC_DATA(weightInput, ge::GeShape({inputSize, hiddenGateSize}), FORMAT_ND, ge::GeShape({inputSize, hiddenGateSize}),
            FORMAT_ND, DT_FLOAT16);
  DESC_DATA(weightHidden, ge::GeShape({tSize, hiddenSize, hiddenGateSize}), FORMAT_ND,
            ge::GeShape({tSize, hiddenSize, hiddenGateSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(weightAtt, ge::GeShape({tSize, batchSize}), FORMAT_ND, ge::GeShape({tSize, batchSize}), FORMAT_FRACTAL_NZ,
            DT_FLOAT16);
  DESC_DATA(biasInput, ge::GeShape({hiddenGateSize}), FORMAT_ND, ge::GeShape({hiddenGateSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(biasHidden, ge::GeShape({hiddenGateSize}), FORMAT_ND, ge::GeShape({hiddenGateSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(seqLen, ge::GeShape({batchSize}), FORMAT_ND, ge::GeShape({batchSize}), FORMAT_ND, DT_INT32);
  DESC_DATA(initH, ge::GeShape({batchSize, hiddenSize}), FORMAT_FRACTAL_NZ, ge::GeShape({batchSize, hiddenSize}),
            FORMAT_FRACTAL_NZ, DT_FLOAT16);
  DESC_DATA(output_y, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_h, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_update, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_updateatt, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_reset, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_new, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_hiddennew, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);

  DESC_DATA(output_rnnGenMask, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_tanh, ge::GeShape({batchSize}), FORMAT_ND, ge::GeShape({batchSize}), FORMAT_ND, DT_FLOAT16);

  ge::OpDescPtr x = std::make_shared<ge::OpDesc>("data_xVec", "Data");
  ge::OpDescPtr weightInput = std::make_shared<ge::OpDesc>("data_weightInput", "Data");
  ge::OpDescPtr weightHidden = std::make_shared<ge::OpDesc>("data_weightHidden", "Data");
  ge::OpDescPtr weightAtt = std::make_shared<ge::OpDesc>("data_weightAtt", "Data");
  ge::OpDescPtr biasInput = std::make_shared<ge::OpDesc>("data_biasInput", "Data");
  ge::OpDescPtr biasHidden = std::make_shared<ge::OpDesc>("data_biasHidden", "Data");
  ge::OpDescPtr seqLen = std::make_shared<ge::OpDesc>("data_seqLen", "Data");
  ge::OpDescPtr initH = std::make_shared<ge::OpDesc>("data_initH", "Data");

  ge::OpDescPtr rnnGenMask = std::make_shared<ge::OpDesc>("rnnGenMask", "RnnGenMask");
  ge::OpDescPtr tanh = std::make_shared<ge::OpDesc>("tanh", "Tanh");
  ge::OpDescPtr dynamicAUGRU = std::make_shared<ge::OpDesc>("dynamicAUGRU", "DynamicAUGRU");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

  x->AddOutputDesc(desc_xVec);
  weightInput->AddOutputDesc(desc_weightInput);
  weightHidden->AddOutputDesc(desc_weightHidden);
  weightAtt->AddOutputDesc(desc_weightAtt);
  biasInput->AddOutputDesc(desc_biasInput);
  biasHidden->AddOutputDesc(desc_biasHidden);
  seqLen->AddOutputDesc(desc_seqLen);
  initH->AddOutputDesc(desc_initH);

  rnnGenMask->AddInputDesc("seq_length", desc_seqLen);
  ge::AttrUtils::SetFloat(rnnGenMask, "num_step", tSize);
  ge::AttrUtils::SetFloat(rnnGenMask, "hidden_size", hiddenSize);
  rnnGenMask->AddOutputDesc("seq_mask", desc_output_rnnGenMask);

  // tanh->AddInputDesc("x", desc_output_rnnGenMask);
  // tanh->AddOutputDesc("y", desc_output_tanh);

  dynamicAUGRU->AddInputDesc("x", desc_xVec);
  dynamicAUGRU->AddInputDesc("weight_input", desc_weightInput);
  dynamicAUGRU->AddInputDesc("weight_hidden", desc_weightHidden);
  dynamicAUGRU->AddInputDesc("weight_att", desc_weightAtt);
  dynamicAUGRU->AddInputDesc("bias_input", desc_biasInput);
  dynamicAUGRU->AddInputDesc("bias_hidden", desc_biasHidden);
  dynamicAUGRU->AddInputDesc("seq_length", desc_seqLen);
  dynamicAUGRU->AddInputDesc("init_h", desc_initH);

  dynamicAUGRU->AddOutputDesc("y", desc_output_y);
  dynamicAUGRU->AddOutputDesc("output_h", desc_output_h);
  dynamicAUGRU->AddOutputDesc("update", desc_output_update);
  dynamicAUGRU->AddOutputDesc("update_att", desc_output_updateatt);
  dynamicAUGRU->AddOutputDesc("reset", desc_output_reset);
  dynamicAUGRU->AddOutputDesc("new", desc_output_new);
  dynamicAUGRU->AddOutputDesc("hidden_new", desc_output_hiddennew);

  netoutput->AddInputDesc(desc_output_y);
  netoutput->AddInputDesc(desc_output_h);
  netoutput->AddInputDesc(desc_output_update);
  netoutput->AddInputDesc(desc_output_updateatt);
  netoutput->AddInputDesc(desc_output_reset);
  netoutput->AddInputDesc(desc_output_new);
  netoutput->AddInputDesc(desc_output_hiddennew);
  netoutput->AddInputDesc(desc_output_rnnGenMask);

  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("DynamicAUGRUFusionPass_graph");
  ge::NodePtr x_node = compute_graph_ptr->AddNode(x);
  ge::NodePtr weightInput_node = compute_graph_ptr->AddNode(weightInput);
  ge::NodePtr weightHidden_node = compute_graph_ptr->AddNode(weightHidden);
  ge::NodePtr weightAtt_node = compute_graph_ptr->AddNode(weightAtt);
  ge::NodePtr biasInput_node = compute_graph_ptr->AddNode(biasInput);
  ge::NodePtr biasHidden_node = compute_graph_ptr->AddNode(biasHidden);
  ge::NodePtr seqLen_node = compute_graph_ptr->AddNode(seqLen);
  ge::NodePtr initH_node = compute_graph_ptr->AddNode(initH);

  ge::NodePtr rnnGenMask_node = compute_graph_ptr->AddNode(rnnGenMask);
  ge::NodePtr tanh_node = compute_graph_ptr->AddNode(tanh);
  ge::NodePtr dynamicAUGRU_node = compute_graph_ptr->AddNode(dynamicAUGRU);
  ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

  ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), dynamicAUGRU_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(weightInput_node->GetOutDataAnchor(0), dynamicAUGRU_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(weightHidden_node->GetOutDataAnchor(0), dynamicAUGRU_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(weightAtt_node->GetOutDataAnchor(0), dynamicAUGRU_node->GetInDataAnchor(3));
  ge::GraphUtils::AddEdge(biasInput_node->GetOutDataAnchor(0), dynamicAUGRU_node->GetInDataAnchor(4));
  ge::GraphUtils::AddEdge(biasHidden_node->GetOutDataAnchor(0), dynamicAUGRU_node->GetInDataAnchor(5));
  ge::GraphUtils::AddEdge(seqLen_node->GetOutDataAnchor(0), dynamicAUGRU_node->GetInDataAnchor(6));
  ge::GraphUtils::AddEdge(initH_node->GetOutDataAnchor(0), dynamicAUGRU_node->GetInDataAnchor(7));

  ge::GraphUtils::AddEdge(dynamicAUGRU_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(dynamicAUGRU_node->GetOutDataAnchor(1), netoutput_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(dynamicAUGRU_node->GetOutDataAnchor(2), netoutput_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(dynamicAUGRU_node->GetOutDataAnchor(3), netoutput_node->GetInDataAnchor(3));
  ge::GraphUtils::AddEdge(dynamicAUGRU_node->GetOutDataAnchor(4), netoutput_node->GetInDataAnchor(4));
  ge::GraphUtils::AddEdge(dynamicAUGRU_node->GetOutDataAnchor(5), netoutput_node->GetInDataAnchor(5));
  ge::GraphUtils::AddEdge(dynamicAUGRU_node->GetOutDataAnchor(6), netoutput_node->GetInDataAnchor(6));

  ge::GraphUtils::AddEdge(seqLen_node->GetOutDataAnchor(0), rnnGenMask_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(rnnGenMask_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(7));

  fe::Status status = fe::FusionPassTestUtils::RunGraphFusionPass("DynamicAUGRUAddSeqPass", fe::BUILT_IN_GRAPH_PASS,
                                                                  *compute_graph_ptr);

  EXPECT_EQ(status, fe::SUCCESS);
}