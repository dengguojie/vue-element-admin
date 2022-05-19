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

class dynamic_rnn_v2_seqlength_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", true);
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", true);
    std::cout << "dynamic_rnn_v2_seqlength_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dynamic_rnn_v2_seqlength_fusion_test TearDown" << std::endl;
  }
};


TEST_F(dynamic_rnn_v2_seqlength_fusion_test, dynamic_rnn_v2_seqlength_fusion_test_1) {
  ge::Graph graph("dynamic_rnn_v2_seqlength_fusion_test_1");
  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{8,1,16};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_weight_input = op::Data("weight_input");
  std::vector<int64_t> data_weight_input_vec{16,4};
  ge::Shape data_weight_input_shape(data_weight_input_vec);
  ge::TensorDesc data_weight_input_desc(data_weight_input_shape, FORMAT_ND, DT_FLOAT);
  data_weight_input.update_input_desc_x(data_weight_input_desc);
  data_weight_input.update_output_desc_y(data_weight_input_desc);

  auto data_weight_hidden = op::Data("weight_hidden");
  std::vector<int64_t> data_weight_hidden_vec{1,4};
  ge::Shape data_weight_hidden_shape(data_weight_hidden_vec);
  ge::TensorDesc data_weight_hidden_desc(data_weight_hidden_shape, FORMAT_ND, DT_FLOAT);
  data_weight_hidden.update_input_desc_x(data_weight_hidden_desc);
  data_weight_hidden.update_output_desc_y(data_weight_hidden_desc);

  // bias_input
  auto data_bias_input = op::Data("b");
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
  auto rnn_op = op::DynamicRNNV2("DynamicRNNV2");
  rnn_op.set_input_x(data_x)
        .set_input_weight_input(data_weight_input)
        .set_input_weight_hidden(data_weight_hidden)
        .set_input_b(data_bias_input)
        .set_input_seq_length(data_seq_length)
        .set_input_init_h(data_init_h)
        .set_input_init_c(data_init_c);
  std::vector<Operator> inputs{data_x, data_weight_input, data_weight_hidden, data_bias_input, data_seq_length, data_init_h, data_init_c};
  std::vector<Operator> outputs{rnn_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicRNNV2SeqFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  
  bool find_gen_mask = false;
  bool find_rnn = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "RnnGenMask" || node->GetType() == "RnnGenMaskV2") {
      find_gen_mask = true;
    }

    if (node->GetType() == "DynamicRNNV2") {
      find_rnn = true;
    }
  }
  EXPECT_EQ(find_gen_mask, true);
}

TEST_F(dynamic_rnn_v2_seqlength_fusion_test, dynamic_rnn_v2_seqlength_fusion_test_2) {
  ge::Graph graph("dynamic_rnn_v2_seqlength_fusion_test_2");
  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{8,1,16};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_weight_input = op::Data("weight_input");
  std::vector<int64_t> data_weight_input_vec{16,4};
  ge::Shape data_weight_input_shape(data_weight_input_vec);
  ge::TensorDesc data_weight_input_desc(data_weight_input_shape, FORMAT_ND, DT_FLOAT);
  data_weight_input.update_input_desc_x(data_weight_input_desc);
  data_weight_input.update_output_desc_y(data_weight_input_desc);

  auto data_weight_hidden = op::Data("weight_hidden");
  std::vector<int64_t> data_weight_hidden_vec{1,4};
  ge::Shape data_weight_hidden_shape(data_weight_hidden_vec);
  ge::TensorDesc data_weight_hidden_desc(data_weight_hidden_shape, FORMAT_ND, DT_FLOAT);
  data_weight_hidden.update_input_desc_x(data_weight_hidden_desc);
  data_weight_hidden.update_output_desc_y(data_weight_hidden_desc);

  // bias_input
  auto data_bias_input = op::Data("b");
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
  auto rnn_op = op::DynamicRNNV2("DynamicRNNV2");
  rnn_op.set_input_x(data_x)
        .set_input_weight_input(data_weight_input)
        .set_input_weight_hidden(data_weight_hidden)
        .set_input_seq_length(data_seq_length)
        .set_input_init_h(data_init_h)
        .set_input_init_c(data_init_c);
  std::vector<Operator> inputs{data_x, data_weight_input, data_weight_hidden, data_bias_input, data_seq_length, data_init_h, data_init_c};
  std::vector<Operator> outputs{rnn_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicRNNV2SeqFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  
  bool find_gen_mask = false;
  bool find_rnn = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "RnnGenMask" || node->GetType() == "RnnGenMaskV2") {
      find_gen_mask = true;
    }

    if (node->GetType() == "DynamicRNNV2") {
      find_rnn = true;
    }
  }
  EXPECT_EQ(find_gen_mask, true);
}

TEST_F(dynamic_rnn_v2_seqlength_fusion_test, dynamic_rnn_v2_seqlength_fusion_test_3) {
  ge::Graph graph("dynamic_rnn_v2_seqlength_fusion_test_3");
  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{-1,1,16};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_weight_input = op::Data("weight_input");
  std::vector<int64_t> data_weight_input_vec{16,4};
  ge::Shape data_weight_input_shape(data_weight_input_vec);
  ge::TensorDesc data_weight_input_desc(data_weight_input_shape, FORMAT_ND, DT_FLOAT);
  data_weight_input.update_input_desc_x(data_weight_input_desc);
  data_weight_input.update_output_desc_y(data_weight_input_desc);

  auto data_weight_hidden = op::Data("weight_hidden");
  std::vector<int64_t> data_weight_hidden_vec{1,4};
  ge::Shape data_weight_hidden_shape(data_weight_hidden_vec);
  ge::TensorDesc data_weight_hidden_desc(data_weight_hidden_shape, FORMAT_ND, DT_FLOAT);
  data_weight_hidden.update_input_desc_x(data_weight_hidden_desc);
  data_weight_hidden.update_output_desc_y(data_weight_hidden_desc);

  // bias_input
  auto data_bias_input = op::Data("b");
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
  auto rnn_op = op::DynamicRNNV2("DynamicRNNV2");
  rnn_op.set_input_x(data_x)
        .set_input_weight_input(data_weight_input)
        .set_input_weight_hidden(data_weight_hidden)
        .set_input_b(data_bias_input)
        .set_input_seq_length(data_seq_length)
        .set_input_init_h(data_init_h)
        .set_input_init_c(data_init_c);
  std::vector<Operator> inputs{data_x, data_weight_input, data_weight_hidden, data_bias_input, data_seq_length, data_init_h, data_init_c};
  std::vector<Operator> outputs{rnn_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicRNNV2SeqFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool find_gen_mask = false;
  bool find_rnn = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "RnnGenMask" || node->GetType() == "RnnGenMaskV2") {
      find_gen_mask = true;
    }

    if (node->GetType() == "DynamicRNNV2") {
      find_rnn = true;
    }
  }
  EXPECT_EQ(find_gen_mask, true);
}

TEST_F(dynamic_rnn_v2_seqlength_fusion_test, dynamic_rnn_v2_seqlength_fusion_test_4) {
  ge::Graph graph("dynamic_rnn_v2_seqlength_fusion_test_1");
  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{8,1,16};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_weight_input = op::Data("weight_input");
  std::vector<int64_t> data_weight_input_vec{16,4};
  ge::Shape data_weight_input_shape(data_weight_input_vec);
  ge::TensorDesc data_weight_input_desc(data_weight_input_shape, FORMAT_ND, DT_FLOAT);
  data_weight_input.update_input_desc_x(data_weight_input_desc);
  data_weight_input.update_output_desc_y(data_weight_input_desc);

  auto data_weight_hidden = op::Data("weight_hidden");
  std::vector<int64_t> data_weight_hidden_vec{1,4};
  ge::Shape data_weight_hidden_shape(data_weight_hidden_vec);
  ge::TensorDesc data_weight_hidden_desc(data_weight_hidden_shape, FORMAT_ND, DT_FLOAT);
  data_weight_hidden.update_input_desc_x(data_weight_hidden_desc);
  data_weight_hidden.update_output_desc_y(data_weight_hidden_desc);

  // bias_input
  auto data_bias_input = op::Data("b");
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
  auto rnn_op = op::DynamicRNNV2("DynamicRNNV2");
  rnn_op.SetAttr("is_misplaced", true);
  rnn_op.set_input_x(data_x)
        .set_input_weight_input(data_weight_input)
        .set_input_weight_hidden(data_weight_hidden)
        .set_input_b(data_bias_input)
        .set_input_seq_length(data_init_h)
        .set_input_init_h(data_init_c);
  std::vector<Operator> inputs{data_x, data_weight_input, data_weight_hidden, data_bias_input, data_init_h, data_init_c};
  std::vector<Operator> outputs{rnn_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicRNNV2SeqFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
}

#define DESC_DATA(name, shape_in, format_in, shape_out, format_out, dtype) \
  ge::GeTensorDesc desc_##name(shape_out, format_out, dtype);              \
  desc_##name.SetOriginFormat(format_in);                                  \
  desc_##name.SetOriginShape(shape_in)

TEST_F(dynamic_rnn_v2_seqlength_fusion_test, dynamic_rnn_v2_seqlength_fusion_test_5) {
  ge::Graph graph("dynamic_rnn_v2_seqlength_fusion_test_5");

  int64_t tSize = 1;
  int64_t batchSize = 128;
  int64_t inputSize = 64;
  int64_t hiddenSize = 32;
  int64_t hiddenGateSize = 4 * hiddenSize;

  DESC_DATA(xVec, ge::GeShape({tSize, batchSize, inputSize}), FORMAT_ND, ge::GeShape({tSize, batchSize, inputSize}),
            FORMAT_ND, DT_FLOAT16);
  DESC_DATA(weightInput, ge::GeShape({inputSize, hiddenGateSize}), FORMAT_ND, ge::GeShape({inputSize, hiddenGateSize}),
            FORMAT_ND, DT_FLOAT16);
  DESC_DATA(weightHidden, ge::GeShape({hiddenSize, hiddenGateSize}), FORMAT_ND,
            ge::GeShape({hiddenSize, hiddenGateSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(bias, ge::GeShape({hiddenGateSize}), FORMAT_ND, ge::GeShape({hiddenGateSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(seqLen, ge::GeShape({batchSize}), FORMAT_ND, ge::GeShape({batchSize}), FORMAT_ND, DT_INT32);

  DESC_DATA(output_y, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_h, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(output_c, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(i, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(j, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(f, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(o, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);
  DESC_DATA(tanhc, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);

  DESC_DATA(output_rnnGenMask, ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND,
            ge::GeShape({tSize, batchSize, hiddenSize}), FORMAT_ND, DT_FLOAT16);

  ge::OpDescPtr x = std::make_shared<ge::OpDesc>("data_xVec", "Data");
  ge::OpDescPtr weightInput = std::make_shared<ge::OpDesc>("data_weightInput", "Data");
  ge::OpDescPtr weightHidden = std::make_shared<ge::OpDesc>("data_weightHidden", "Data");
  ge::OpDescPtr bias = std::make_shared<ge::OpDesc>("data_bias", "Data");
  ge::OpDescPtr seqLen = std::make_shared<ge::OpDesc>("data_seqLen", "Data");

  ge::OpDescPtr rnnGenMask = std::make_shared<ge::OpDesc>("rnnGenMask", "RnnGenMask");
  ge::OpDescPtr dynamicRNNV2 = std::make_shared<ge::OpDesc>("dynamicRNNV2", "DynamicRNNV2");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

  x->AddOutputDesc(desc_xVec);
  weightInput->AddOutputDesc(desc_weightInput);
  weightHidden->AddOutputDesc(desc_weightHidden);
  bias->AddOutputDesc(desc_bias);
  seqLen->AddOutputDesc(desc_seqLen);

  rnnGenMask->AddInputDesc("seq_length", desc_seqLen);
  ge::AttrUtils::SetFloat(rnnGenMask, "num_step", tSize);
  ge::AttrUtils::SetFloat(rnnGenMask, "hidden_size", hiddenSize);
  rnnGenMask->AddOutputDesc("seq_mask", desc_output_rnnGenMask);

  dynamicRNNV2->AddInputDesc("x", desc_xVec);
  dynamicRNNV2->AddInputDesc("weight_input", desc_weightInput);
  dynamicRNNV2->AddInputDesc("weight_hidden", desc_weightHidden);
  dynamicRNNV2->AddInputDesc("b", desc_bias);
  dynamicRNNV2->AddInputDesc("seq_length", desc_seqLen);

  dynamicRNNV2->AddOutputDesc("y", desc_output_y);
  dynamicRNNV2->AddOutputDesc("output_h", desc_output_h);
  dynamicRNNV2->AddOutputDesc("output_c", desc_output_c);
  dynamicRNNV2->AddOutputDesc("i", desc_i);
  dynamicRNNV2->AddOutputDesc("j", desc_j);
  dynamicRNNV2->AddOutputDesc("f", desc_f);
  dynamicRNNV2->AddOutputDesc("o", desc_o);
  dynamicRNNV2->AddOutputDesc("tanhc", desc_tanhc);

  netoutput->AddInputDesc(desc_output_y);
  netoutput->AddInputDesc(desc_output_h);
  netoutput->AddInputDesc(desc_output_c);
  netoutput->AddInputDesc(desc_i);
  netoutput->AddInputDesc(desc_j);
  netoutput->AddInputDesc(desc_f);
  netoutput->AddInputDesc(desc_o);
  netoutput->AddInputDesc(desc_tanhc);

  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("DynamicRNNV2FusionPass_graph");
  ge::NodePtr x_node = compute_graph_ptr->AddNode(x);
  ge::NodePtr weightInput_node = compute_graph_ptr->AddNode(weightInput);
  ge::NodePtr weightHidden_node = compute_graph_ptr->AddNode(weightHidden);
  ge::NodePtr bias_node = compute_graph_ptr->AddNode(bias);
  ge::NodePtr seqLen_node = compute_graph_ptr->AddNode(seqLen);

  ge::NodePtr rnnGenMask_node = compute_graph_ptr->AddNode(rnnGenMask);
  ge::NodePtr dynamicRNNV2_node = compute_graph_ptr->AddNode(dynamicRNNV2);
  ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

  ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), dynamicRNNV2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(weightInput_node->GetOutDataAnchor(0), dynamicRNNV2_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(weightHidden_node->GetOutDataAnchor(0), dynamicRNNV2_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(bias_node->GetOutDataAnchor(0), dynamicRNNV2_node->GetInDataAnchor(3));
  ge::GraphUtils::AddEdge(seqLen_node->GetOutDataAnchor(0), dynamicRNNV2_node->GetInDataAnchor(4));
 
  ge::GraphUtils::AddEdge(dynamicRNNV2_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(dynamicRNNV2_node->GetOutDataAnchor(1), netoutput_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(dynamicRNNV2_node->GetOutDataAnchor(2), netoutput_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(dynamicRNNV2_node->GetOutDataAnchor(3), netoutput_node->GetInDataAnchor(3));
  ge::GraphUtils::AddEdge(dynamicRNNV2_node->GetOutDataAnchor(4), netoutput_node->GetInDataAnchor(4));
  ge::GraphUtils::AddEdge(dynamicRNNV2_node->GetOutDataAnchor(5), netoutput_node->GetInDataAnchor(5));
  ge::GraphUtils::AddEdge(dynamicRNNV2_node->GetOutDataAnchor(6), netoutput_node->GetInDataAnchor(6));
  ge::GraphUtils::AddEdge(dynamicRNNV2_node->GetOutDataAnchor(7), netoutput_node->GetInDataAnchor(7));

  ge::GraphUtils::AddEdge(seqLen_node->GetOutDataAnchor(0), rnnGenMask_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(rnnGenMask_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(7));

  fe::Status status = fe::FusionPassTestUtils::RunGraphFusionPass("DynamicRNNV2SeqFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                                                  *compute_graph_ptr);

  EXPECT_EQ(status, fe::SUCCESS);
}
