#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "rnn.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "fp16_t.hpp"
#include "elewise_calculation_ops.h"
#include "nn_norm_ops.h"

using namespace ge;
using namespace op;

class dynamic_rnn_seqlength_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", true);
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", true);
    std::cout << "dynamic_rnn_seqlength_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dynamic_rnn_seqlength_fusion_test TearDown" << std::endl;
  }
};


TEST_F(dynamic_rnn_seqlength_fusion_test, dynamic_rnn_seqlength_fusion_test_1) {
  ge::Graph graph("dynamic_rnn_seqlength_fusion_test_1");
  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{8,1,16};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_weight_input = op::Data("weight_input");
  std::vector<int64_t> data_weight_input_vec{17,4};
  ge::Shape data_weight_input_shape(data_weight_input_vec);
  ge::TensorDesc data_weight_input_desc(data_weight_input_shape, FORMAT_ND, DT_FLOAT);
  data_weight_input.update_input_desc_x(data_weight_input_desc);
  data_weight_input.update_output_desc_y(data_weight_input_desc);

  // bias_input
  auto data_bias_input = op::Data("bias_input");
  std::vector<int64_t> data_bias_input_vec{4};
  ge::Shape data_bias_input_shape(data_bias_input_vec);
  ge::TensorDesc data_bias_input_desc(data_bias_input_shape, FORMAT_ND, DT_FLOAT);
  data_bias_input.update_input_desc_x(data_bias_input_desc);
  data_bias_input.update_output_desc_y(data_bias_input_desc);

  // seq_length
  auto data_seq_length = op::Data("seq_length");
  std::vector<int64_t> data_seq_length_vec{1};
  ge::Shape data_seq_length_shape(data_seq_length_vec);
  ge::TensorDesc data_seq_length_desc(data_seq_length_shape, FORMAT_ND, DT_INT32);
  data_seq_length.update_input_desc_x(data_seq_length_desc);
  data_seq_length.update_output_desc_y(data_seq_length_desc);

  // init_h
  auto data_init_h = op::Data("init_h");
  std::vector<int64_t> data_init_h_vec{1,1};
  ge::Shape data_init_h_shape(data_init_h_vec);
  ge::TensorDesc data_init_h_desc(data_init_h_shape, FORMAT_ND, DT_FLOAT);
  data_init_h.update_input_desc_x(data_init_h_desc);
  data_init_h.update_output_desc_y(data_init_h_desc);

  // init_c
  auto data_init_c = op::Data("init_c");
  std::vector<int64_t> data_init_c_vec{1,1};
  ge::Shape data_init_c_shape(data_init_c_vec);
  ge::TensorDesc data_init_c_desc(data_init_c_shape, FORMAT_ND, DT_FLOAT);
  data_init_c.update_input_desc_x(data_init_c_desc);
  data_init_c.update_output_desc_y(data_init_c_desc);
  auto rnn_op = op::DynamicRNN("DynamicRNN");
  rnn_op.set_input_x(data_x)
        .set_input_w(data_weight_input)
        .set_input_b(data_bias_input)
        .set_input_seq_length(data_seq_length)
        .set_input_init_h(data_init_h)
        .set_input_init_c(data_init_c);
  std::vector<Operator> inputs{data_x, data_weight_input, data_bias_input, data_seq_length, data_init_h, data_init_c};
  std::vector<Operator> outputs{rnn_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicRNNSeqFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  
  bool find_gen_mask = false;
  bool find_rnn = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "RnnGenMask" || node->GetType() == "RnnGenMaskV2") {
      find_gen_mask = true;
    }

    if (node->GetType() == "DynamicRNN") {
      find_rnn = true;
    }
  }
  EXPECT_EQ(find_gen_mask, true);
}

TEST_F(dynamic_rnn_seqlength_fusion_test, dynamic_rnn_seqlength_fusion_test_2) {
  ge::Graph graph("dynamic_rnn_seqlength_fusion_test_2");
  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{8,1,16};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_weight_input = op::Data("weight_input");
  std::vector<int64_t> data_weight_input_vec{17,4};
  ge::Shape data_weight_input_shape(data_weight_input_vec);
  ge::TensorDesc data_weight_input_desc(data_weight_input_shape, FORMAT_ND, DT_FLOAT);
  data_weight_input.update_input_desc_x(data_weight_input_desc);
  data_weight_input.update_output_desc_y(data_weight_input_desc);

  // bias_input
  auto data_bias_input = op::Data("bias_input");
  std::vector<int64_t> data_bias_input_vec{4};
  ge::Shape data_bias_input_shape(data_bias_input_vec);
  ge::TensorDesc data_bias_input_desc(data_bias_input_shape, FORMAT_ND, DT_FLOAT);
  data_bias_input.update_input_desc_x(data_bias_input_desc);
  data_bias_input.update_output_desc_y(data_bias_input_desc);

  // seq_length
  auto data_seq_length = op::Data("seq_length");
  std::vector<int64_t> data_seq_length_vec;
  ge::Shape data_seq_length_shape(data_seq_length_vec);
  ge::TensorDesc data_seq_length_desc(data_seq_length_shape, FORMAT_ND, DT_INT32);
  data_seq_length.update_input_desc_x(data_seq_length_desc);
  data_seq_length.update_output_desc_y(data_seq_length_desc);

  // init_h
  auto data_init_h = op::Data("init_h");
  std::vector<int64_t> data_init_h_vec{1,1};
  ge::Shape data_init_h_shape(data_init_h_vec);
  ge::TensorDesc data_init_h_desc(data_init_h_shape, FORMAT_ND, DT_FLOAT);
  data_init_h.update_input_desc_x(data_init_h_desc);
  data_init_h.update_output_desc_y(data_init_h_desc);

  // init_c
  auto data_init_c = op::Data("init_c");
  std::vector<int64_t> data_init_c_vec{1,1};
  ge::Shape data_init_c_shape(data_init_c_vec);
  ge::TensorDesc data_init_c_desc(data_init_c_shape, FORMAT_ND, DT_FLOAT);
  data_init_c.update_input_desc_x(data_init_c_desc);
  data_init_c.update_output_desc_y(data_init_c_desc);
  auto rnn_op = op::DynamicRNN("DynamicRNN");
  rnn_op.set_input_x(data_x)
        .set_input_w(data_weight_input)
        .set_input_b(data_bias_input)
        .set_input_seq_length(data_seq_length)
        .set_input_init_h(data_init_h)
        .set_input_init_c(data_init_c);
  std::vector<Operator> inputs{data_x, data_weight_input, data_bias_input, data_seq_length, data_init_h, data_init_c};
  std::vector<Operator> outputs{rnn_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicRNNSeqFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  
  bool find_gen_mask = false;
  bool find_rnn = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "RnnGenMask" || node->GetType() == "RnnGenMaskV2") {
      find_gen_mask = true;
    }

    if (node->GetType() == "DynamicRNN") {
      find_rnn = true;
    }
  }
  EXPECT_EQ(find_gen_mask, true);
}

TEST_F(dynamic_rnn_seqlength_fusion_test, dynamic_rnn_seqlength_fusion_test_3) {
  ge::Graph graph("dynamic_rnn_seqlength_fusion_test_3");
  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{-1,1,16};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_weight_input = op::Data("weight_input");
  std::vector<int64_t> data_weight_input_vec{17,4};
  ge::Shape data_weight_input_shape(data_weight_input_vec);
  ge::TensorDesc data_weight_input_desc(data_weight_input_shape, FORMAT_ND, DT_FLOAT);
  data_weight_input.update_input_desc_x(data_weight_input_desc);
  data_weight_input.update_output_desc_y(data_weight_input_desc);

  // bias_input
  auto data_bias_input = op::Data("bias_input");
  std::vector<int64_t> data_bias_input_vec{4};
  ge::Shape data_bias_input_shape(data_bias_input_vec);
  ge::TensorDesc data_bias_input_desc(data_bias_input_shape, FORMAT_ND, DT_FLOAT);
  data_bias_input.update_input_desc_x(data_bias_input_desc);
  data_bias_input.update_output_desc_y(data_bias_input_desc);

  // seq_length
  auto data_seq_length = op::Data("seq_length");
  std::vector<int64_t> data_seq_length_vec{1};
  ge::Shape data_seq_length_shape(data_seq_length_vec);
  ge::TensorDesc data_seq_length_desc(data_seq_length_shape, FORMAT_ND, DT_INT32);
  data_seq_length.update_input_desc_x(data_seq_length_desc);
  data_seq_length.update_output_desc_y(data_seq_length_desc);

  // init_h
  auto data_init_h = op::Data("init_h");
  std::vector<int64_t> data_init_h_vec{1,1};
  ge::Shape data_init_h_shape(data_init_h_vec);
  ge::TensorDesc data_init_h_desc(data_init_h_shape, FORMAT_ND, DT_FLOAT);
  data_init_h.update_input_desc_x(data_init_h_desc);
  data_init_h.update_output_desc_y(data_init_h_desc);

  // init_c
  auto data_init_c = op::Data("init_c");
  std::vector<int64_t> data_init_c_vec{1,1};
  ge::Shape data_init_c_shape(data_init_c_vec);
  ge::TensorDesc data_init_c_desc(data_init_c_shape, FORMAT_ND, DT_FLOAT);
  data_init_c.update_input_desc_x(data_init_c_desc);
  data_init_c.update_output_desc_y(data_init_c_desc);
  auto rnn_op = op::DynamicRNN("DynamicRNN");
  rnn_op.set_input_x(data_x)
        .set_input_w(data_weight_input)
        .set_input_b(data_bias_input)
        .set_input_seq_length(data_seq_length)
        .set_input_init_h(data_init_h)
        .set_input_init_c(data_init_c);
  std::vector<Operator> inputs{data_x, data_weight_input, data_bias_input, data_seq_length, data_init_h, data_init_c};
  std::vector<Operator> outputs{rnn_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicRNNSeqFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool find_gen_mask = false;
  bool find_rnn = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "RnnGenMask" || node->GetType() == "RnnGenMaskV2") {
      find_gen_mask = true;
    }

    if (node->GetType() == "DynamicRNN") {
      find_rnn = true;
    }
  }
  EXPECT_EQ(find_gen_mask, true);
}

TEST_F(dynamic_rnn_seqlength_fusion_test, dynamic_rnn_seqlength_fusion_test_4) {
  ge::Graph graph("dynamic_rnn_seqlength_fusion_test_1");
  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{8,1,16};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_weight_input = op::Data("weight_input");
  std::vector<int64_t> data_weight_input_vec{17,4};
  ge::Shape data_weight_input_shape(data_weight_input_vec);
  ge::TensorDesc data_weight_input_desc(data_weight_input_shape, FORMAT_ND, DT_FLOAT);
  data_weight_input.update_input_desc_x(data_weight_input_desc);
  data_weight_input.update_output_desc_y(data_weight_input_desc);

  // bias_input
  auto data_bias_input = op::Data("bias_input");
  std::vector<int64_t> data_bias_input_vec{4};
  ge::Shape data_bias_input_shape(data_bias_input_vec);
  ge::TensorDesc data_bias_input_desc(data_bias_input_shape, FORMAT_ND, DT_FLOAT);
  data_bias_input.update_input_desc_x(data_bias_input_desc);
  data_bias_input.update_output_desc_y(data_bias_input_desc);

  // seq_length
  auto data_seq_length = op::Data("seq_length");
  std::vector<int64_t> data_seq_length_vec{1};
  ge::Shape data_seq_length_shape(data_seq_length_vec);
  ge::TensorDesc data_seq_length_desc(data_seq_length_shape, FORMAT_ND, DT_INT32);
  data_seq_length.update_input_desc_x(data_seq_length_desc);
  data_seq_length.update_output_desc_y(data_seq_length_desc);

  // init_h
  auto data_init_h = op::Data("init_h");
  std::vector<int64_t> data_init_h_vec{1,1};
  ge::Shape data_init_h_shape(data_init_h_vec);
  ge::TensorDesc data_init_h_desc(data_init_h_shape, FORMAT_ND, DT_FLOAT);
  data_init_h.update_input_desc_x(data_init_h_desc);
  data_init_h.update_output_desc_y(data_init_h_desc);

  // init_c
  auto data_init_c = op::Data("init_c");
  std::vector<int64_t> data_init_c_vec{1,1};
  ge::Shape data_init_c_shape(data_init_c_vec);
  ge::TensorDesc data_init_c_desc(data_init_c_shape, FORMAT_ND, DT_FLOAT);
  data_init_c.update_input_desc_x(data_init_c_desc);
  data_init_c.update_output_desc_y(data_init_c_desc);
  auto rnn_op = op::DynamicRNN("DynamicRNN");
  rnn_op.SetAttr("is_misplaced", true);
  rnn_op.set_input_x(data_x)
        .set_input_w(data_weight_input)
        .set_input_b(data_bias_input)
        .set_input_seq_length(data_init_h)
        .set_input_init_h(data_init_c);
  std::vector<Operator> inputs{data_x, data_weight_input, data_bias_input, data_init_h, data_init_c};
  std::vector<Operator> outputs{rnn_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicRNNSeqFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
}

#define DESC_DATA(name, shape_in, format_in, shape_out, format_out, dtype) \
  ge::GeTensorDesc desc_##name(shape_out, format_out, dtype);              \
  desc_##name.SetOriginFormat(format_in);                                  \
  desc_##name.SetOriginShape(shape_in)

TEST_F(dynamic_rnn_seqlength_fusion_test, dynamic_rnn_seqlength_fusion_test_5) {
  ge::Graph graph("dynamic_rnn_seqlength_fusion_test_5");

  DESC_DATA(xVec, ge::GeShape({8, 1, 16}), FORMAT_ND, ge::GeShape({8, 1, 16}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(wVec, ge::GeShape({17, 4}), FORMAT_ND, ge::GeShape({17, 4}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(biasInput, ge::GeShape({4}), FORMAT_ND, ge::GeShape({4}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(seqLen, ge::GeShape({1}), FORMAT_ND, ge::GeShape({1}), FORMAT_ND, DT_INT32);
  DESC_DATA(initH, ge::GeShape({1, 1}), FORMAT_ND, ge::GeShape({1, 1}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(initC, ge::GeShape({1, 1}), FORMAT_ND, ge::GeShape({1, 1}), FORMAT_ND, DT_FLOAT16);

  DESC_DATA(output_y, ge::GeShape({8, 1, 1}), FORMAT_ND, ge::GeShape({8, 1, 1}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_h, ge::GeShape({8, 1, 1}), FORMAT_ND, ge::GeShape({8, 1, 1}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_c, ge::GeShape({8, 1, 1}), FORMAT_ND, ge::GeShape({8, 1, 1}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_i, ge::GeShape({8, 1, 1}), FORMAT_ND, ge::GeShape({8, 1, 1}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_j, ge::GeShape({8, 1, 1}), FORMAT_ND, ge::GeShape({8, 1, 1}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_f, ge::GeShape({8, 1, 1}), FORMAT_ND, ge::GeShape({8, 1, 1}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_o, ge::GeShape({8, 1, 1}), FORMAT_ND, ge::GeShape({8, 1, 1}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_tanhc, ge::GeShape({8, 1, 1}), FORMAT_ND, ge::GeShape({8, 1, 1}), FORMAT_ND, DT_FLOAT16);

  DESC_DATA(output_rnnGenMask, ge::GeShape({8, 1, 1}), FORMAT_ND, ge::GeShape({8, 1, 1}), FORMAT_ND, DT_FLOAT16);

  ge::OpDescPtr x = std::make_shared<ge::OpDesc>("xVec", "Data");
  ge::OpDescPtr w = std::make_shared<ge::OpDesc>("wVec", "Data");
  ge::OpDescPtr biasInput = std::make_shared<ge::OpDesc>("biasInput", "Data");
  ge::OpDescPtr seqLen = std::make_shared<ge::OpDesc>("seqLen", "Data");
  ge::OpDescPtr initH = std::make_shared<ge::OpDesc>("initH", "Data");
  ge::OpDescPtr initC = std::make_shared<ge::OpDesc>("initC", "Data");

  ge::OpDescPtr rnnGenMask = std::make_shared<ge::OpDesc>("rnnGenMask", "RnnGenMask");
  ge::OpDescPtr dynamicRNN = std::make_shared<ge::OpDesc>("dynamicRNN", "DynamicRNN");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

  x->AddOutputDesc(desc_xVec);
  w->AddOutputDesc(desc_wVec);
  biasInput->AddOutputDesc(desc_biasInput);
  seqLen->AddOutputDesc(desc_seqLen);
  initH->AddOutputDesc(desc_initH);
  initC->AddOutputDesc(desc_initC);

  rnnGenMask->AddInputDesc("seq_length", desc_seqLen);
  ge::AttrUtils::SetFloat(rnnGenMask, "num_step", 8);
  ge::AttrUtils::SetFloat(rnnGenMask, "hidden_size", 1);
  rnnGenMask->AddOutputDesc("seq_mask", desc_output_rnnGenMask);

  dynamicRNN->AddInputDesc("x", desc_xVec);
  dynamicRNN->AddInputDesc("w", desc_wVec);
  dynamicRNN->AddInputDesc("b", desc_biasInput);
  dynamicRNN->AddInputDesc("seq_length", desc_seqLen);
  dynamicRNN->AddInputDesc("init_h", desc_initH);
  dynamicRNN->AddInputDesc("init_c", desc_initC);

  dynamicRNN->AddOutputDesc("y", desc_output_y);
  dynamicRNN->AddOutputDesc("output_h", desc_output_h);
  dynamicRNN->AddOutputDesc("output_c", desc_output_c);
  dynamicRNN->AddOutputDesc("i", desc_output_i);
  dynamicRNN->AddOutputDesc("j", desc_output_j);
  dynamicRNN->AddOutputDesc("f", desc_output_f);
  dynamicRNN->AddOutputDesc("o", desc_output_o);
  dynamicRNN->AddOutputDesc("tanhc", desc_output_tanhc);

  netoutput->AddInputDesc(desc_output_y);
  netoutput->AddInputDesc(desc_output_h);
  netoutput->AddInputDesc(desc_output_c);
  netoutput->AddInputDesc(desc_output_i);
  netoutput->AddInputDesc(desc_output_j);
  netoutput->AddInputDesc(desc_output_f);
  netoutput->AddInputDesc(desc_output_o);
  netoutput->AddInputDesc(desc_output_tanhc);
  netoutput->AddInputDesc(desc_output_rnnGenMask);

  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("dynamicRNNFusionPass_graph");
  ge::NodePtr x_node = compute_graph_ptr->AddNode(x);
  ge::NodePtr w_node = compute_graph_ptr->AddNode(w);
  ge::NodePtr biasInput_node = compute_graph_ptr->AddNode(biasInput);
  ge::NodePtr seqLen_node = compute_graph_ptr->AddNode(seqLen);
  ge::NodePtr initH_node = compute_graph_ptr->AddNode(initH);
  ge::NodePtr initC_node = compute_graph_ptr->AddNode(initC);

  ge::NodePtr rnnGenMask_node = compute_graph_ptr->AddNode(rnnGenMask);
  ge::NodePtr dynamicRNN_node = compute_graph_ptr->AddNode(dynamicRNN);
  ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

  ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), dynamicRNN_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(w_node->GetOutDataAnchor(0), dynamicRNN_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(biasInput_node->GetOutDataAnchor(0), dynamicRNN_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(seqLen_node->GetOutDataAnchor(0), dynamicRNN_node->GetInDataAnchor(3));
  ge::GraphUtils::AddEdge(initH_node->GetOutDataAnchor(0), dynamicRNN_node->GetInDataAnchor(4));
  ge::GraphUtils::AddEdge(initC_node->GetOutDataAnchor(0), dynamicRNN_node->GetInDataAnchor(5));

  ge::GraphUtils::AddEdge(dynamicRNN_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(dynamicRNN_node->GetOutDataAnchor(1), netoutput_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(dynamicRNN_node->GetOutDataAnchor(2), netoutput_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(dynamicRNN_node->GetOutDataAnchor(3), netoutput_node->GetInDataAnchor(3));
  ge::GraphUtils::AddEdge(dynamicRNN_node->GetOutDataAnchor(4), netoutput_node->GetInDataAnchor(4));
  ge::GraphUtils::AddEdge(dynamicRNN_node->GetOutDataAnchor(5), netoutput_node->GetInDataAnchor(5));
  ge::GraphUtils::AddEdge(dynamicRNN_node->GetOutDataAnchor(6), netoutput_node->GetInDataAnchor(6));
  ge::GraphUtils::AddEdge(dynamicRNN_node->GetOutDataAnchor(7), netoutput_node->GetInDataAnchor(7));

  ge::GraphUtils::AddEdge(seqLen_node->GetOutDataAnchor(0), rnnGenMask_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(rnnGenMask_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(8));

  fe::Status status = fe::FusionPassTestUtils::RunGraphFusionPass("DynamicRNNSeqFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                                                  *compute_graph_ptr);

  EXPECT_EQ(status, fe::SUCCESS);
}