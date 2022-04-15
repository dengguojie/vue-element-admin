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