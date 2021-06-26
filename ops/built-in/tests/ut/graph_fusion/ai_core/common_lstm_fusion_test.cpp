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

class common_lstm_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "common_lstm_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "common_lstm_fusion_test TearDown" << std::endl;
  }
};

TEST_F(common_lstm_fusion_test, common_lstm_fusion_test_1) {
  ge::Graph graph("common_lstm_fusion_test_1");

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{20,2,160};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_w = op::Const("w");
  std::vector<int64_t> data_w_vec{1,32*4,160};
  ge::Shape data_w_shape(data_w_vec);
  ge::TensorDesc data_w_desc(data_w_shape, FORMAT_ND, DT_FLOAT);
  Tensor w_tensor;
  float * w_tensor_value = new float[160*32*4];
  w_tensor.SetTensorDesc(data_w_desc);
  w_tensor.SetData((uint8_t*)w_tensor_value, 160*32*4*sizeof(float));
  data_w.set_attr_value(w_tensor);
  data_w.update_output_desc_y(data_w_desc);
  
  // r
  auto data_r = op::Const("r");
  std::vector<int64_t> data_r_vec{1,32*4,32};
  ge::Shape data_r_shape(data_r_vec);
  ge::TensorDesc data_r_desc(data_r_shape, FORMAT_ND, DT_FLOAT);
  Tensor r_tensor;
  float * r_tensor_value = new float[32*32*4];
  r_tensor.SetTensorDesc(data_r_desc);
  r_tensor.SetData((uint8_t*)r_tensor_value, 32*32*4*sizeof(float));
  data_r.set_attr_value(r_tensor);
  data_r.update_output_desc_y(data_r_desc);

  // b
  auto data_b = op::Const("b");
  std::vector<int64_t> data_b_vec{1,32*4*2};
  ge::Shape data_b_shape(data_b_vec);
  ge::TensorDesc data_b_desc(data_b_shape, FORMAT_ND, DT_FLOAT);
  Tensor b_tensor;
  float * b_tensor_value = new float[2*32*4];
  b_tensor.SetTensorDesc(data_b_desc);
  b_tensor.SetData((uint8_t*)b_tensor_value, 2*32*4*sizeof(float));
  data_b.set_attr_value(b_tensor);
  data_b.update_output_desc_y(data_b_desc);

  auto common_lstm_op = op::CommonLSTM("CommonLSTM");
  common_lstm_op.set_input_x(data_x)
           .set_input_w(data_w)
           .set_input_r(data_r)
           .set_input_b(data_b)
           .set_attr_hidden_size(32);

  std::vector<Operator> inputs{data_x};
  std::vector<Operator> outputs{common_lstm_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraphPtr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(computeGraphPtr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CommonLSTMFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraphPtr);

  bool findDynamicRNN = false;
  bool findReshape = false;

  for (auto node: computeGraphPtr->GetAllNodes()) {
    if (node->GetType() == "DynamicRNN") {
      findDynamicRNN = true;
    }

    if (node->GetType() == "Reshape") {
    findReshape = true;
    }
  }
  EXPECT_EQ(findDynamicRNN, true);
  EXPECT_EQ(findReshape, true);
}

TEST_F(common_lstm_fusion_test, common_lstm_fusion_test_2) {
  ge::Graph graph("common_lstm_fusion_test_2");

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{20,2,160};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_w = op::Const("w");
  std::vector<int64_t> data_w_vec{1,32*4,160};
  ge::Shape data_w_shape(data_w_vec);
  ge::TensorDesc data_w_desc(data_w_shape, FORMAT_ND, DT_FLOAT);
  Tensor w_tensor;
  float * w_tensor_value = new float[160*32*4];
  w_tensor.SetTensorDesc(data_w_desc);
  w_tensor.SetData((uint8_t*)w_tensor_value, 160*32*4*sizeof(float));
  data_w.set_attr_value(w_tensor);
  data_w.update_output_desc_y(data_w_desc);
  
  // r
  auto data_r = op::Const("r");
  std::vector<int64_t> data_r_vec{1,32*4,32};
  ge::Shape data_r_shape(data_r_vec);
  ge::TensorDesc data_r_desc(data_r_shape, FORMAT_ND, DT_FLOAT);
  Tensor r_tensor;
  float * r_tensor_value = new float[32*32*4];
  r_tensor.SetTensorDesc(data_r_desc);
  r_tensor.SetData((uint8_t*)r_tensor_value, 32*32*4*sizeof(float));
  data_r.set_attr_value(r_tensor);
  data_r.update_output_desc_y(data_r_desc);


  auto common_lstm_op = op::CommonLSTM("CommonLSTM");
  common_lstm_op.set_input_x(data_x)
           .set_input_w(data_w)
           .set_input_r(data_r)
           .set_attr_hidden_size(32);

  std::vector<Operator> inputs{data_x};
  std::vector<Operator> outputs{common_lstm_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraphPtr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(computeGraphPtr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CommonLSTMFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraphPtr);

  bool findDynamicRNN = false;
  bool findReshape = false;

  for (auto node: computeGraphPtr->GetAllNodes()) {
    if (node->GetType() == "DynamicRNN") {
      findDynamicRNN = true;
    }

    if (node->GetType() == "Reshape") {
    findReshape = true;
    }
  }
  EXPECT_EQ(findDynamicRNN, true);
  EXPECT_EQ(findReshape, true);
}

TEST_F(common_lstm_fusion_test, common_lstm_fusion_test_3) {
  ge::Graph graph("common_lstm_fusion_test_3");

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{1,2,160};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_w = op::Const("w");
  std::vector<int64_t> data_w_vec{1,32*4,160};
  ge::Shape data_w_shape(data_w_vec);
  ge::TensorDesc data_w_desc(data_w_shape, FORMAT_ND, DT_FLOAT);
  Tensor w_tensor;
  float * w_tensor_value = new float[160*32*4];
  w_tensor.SetTensorDesc(data_w_desc);
  w_tensor.SetData((uint8_t*)w_tensor_value, 160*32*4*sizeof(float));
  data_w.set_attr_value(w_tensor);
  data_w.update_output_desc_y(data_w_desc);

  // r
  auto data_r = op::Const("r");
  std::vector<int64_t> data_r_vec{1,32*4,32};
  ge::Shape data_r_shape(data_r_vec);
  ge::TensorDesc data_r_desc(data_r_shape, FORMAT_ND, DT_FLOAT);
  Tensor r_tensor;
  float * r_tensor_value = new float[32*32*4];
  r_tensor.SetTensorDesc(data_r_desc);
  r_tensor.SetData((uint8_t*)r_tensor_value, 32*32*4*sizeof(float));
  data_r.set_attr_value(r_tensor);
  data_r.update_output_desc_y(data_r_desc);
  // b
  auto data_b = op::Const("b");
  std::vector<int64_t> data_b_vec{1,32*4*2};
  ge::Shape data_b_shape(data_b_vec);
  ge::TensorDesc data_b_desc(data_b_shape, FORMAT_ND, DT_FLOAT);
  Tensor b_tensor;
  float * b_tensor_value = new float[2*32*4];
  b_tensor.SetTensorDesc(data_b_desc);
  b_tensor.SetData((uint8_t*)b_tensor_value, 2*32*4*sizeof(float));
  data_b.set_attr_value(b_tensor);
  data_b.update_output_desc_y(data_b_desc);

  // seq_len
  auto data_seq = op::Data("sequence_lens");
  std::vector<int64_t> data_seq_vec{1};
  ge::Shape data_seq_shape(data_seq_vec);
  ge::TensorDesc data_seq_desc(data_seq_shape, FORMAT_ND, DT_FLOAT);
  data_seq.update_input_desc_x(data_seq_desc);
  data_seq.update_output_desc_y(data_seq_desc);

  auto common_lstm_op = op::CommonLSTM("CommonLSTM");
  common_lstm_op.set_input_x(data_x)
  .set_input_w(data_w)
  .set_input_r(data_r)
  .set_input_b(data_b)
  .set_input_sequence_lens(data_seq)
  .set_attr_hidden_size(32);

  std::vector<Operator> inputs{data_x, data_seq};
  std::vector<Operator> outputs{common_lstm_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraphPtr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(computeGraphPtr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CommonLSTMFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraphPtr);

  bool findDynamicRNN = false;
  for (auto node: computeGraphPtr->GetAllNodes()) {
    if (node->GetType() == "DynamicRNN") {
      findDynamicRNN = true;
    }
  }
  EXPECT_EQ(findDynamicRNN, true);
}

TEST_F(common_lstm_fusion_test, common_lstm_fusion_test_4) {
  ge::Graph graph("common_lstm_fusion_test_4");

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{1,2,160};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_w = op::Const("w");
  std::vector<int64_t> data_w_vec{2,32*4,160};
  ge::Shape data_w_shape(data_w_vec);
  ge::TensorDesc data_w_desc(data_w_shape, FORMAT_ND, DT_FLOAT);
  Tensor w_tensor;
  float * w_tensor_value = new float[160*32*4*2];
  w_tensor.SetTensorDesc(data_w_desc);
  w_tensor.SetData((uint8_t*)w_tensor_value, 160*32*4*2*sizeof(float));
  data_w.set_attr_value(w_tensor);
  data_w.update_output_desc_y(data_w_desc);

  // r
  auto data_r = op::Const("r");
  std::vector<int64_t> data_r_vec{2,32*4,32};
  ge::Shape data_r_shape(data_r_vec);
  ge::TensorDesc data_r_desc(data_r_shape, FORMAT_ND, DT_FLOAT);
  Tensor r_tensor;
  float * r_tensor_value = new float[32*32*4*2];
  r_tensor.SetTensorDesc(data_r_desc);
  r_tensor.SetData((uint8_t*)r_tensor_value, 32*32*4*2*sizeof(float));
  data_r.set_attr_value(r_tensor);
  data_r.update_output_desc_y(data_r_desc);
  // b
  auto data_b = op::Const("b");
  std::vector<int64_t> data_b_vec{2,32*4*2};
  ge::Shape data_b_shape(data_b_vec);
  ge::TensorDesc data_b_desc(data_b_shape, FORMAT_ND, DT_FLOAT);
  Tensor b_tensor;
  float * b_tensor_value = new float[2*32*4*2];
  b_tensor.SetTensorDesc(data_b_desc);
  b_tensor.SetData((uint8_t*)b_tensor_value, 2*32*4*2*sizeof(float));
  data_b.set_attr_value(b_tensor);
  data_b.update_output_desc_y(data_b_desc);


  auto common_lstm_op = op::CommonLSTM("CommonLSTM");
  string direction = "bidirectional";
  common_lstm_op.set_input_x(data_x)
  .set_input_w(data_w)
  .set_input_r(data_r)
  .set_input_b(data_b)
  .set_attr_hidden_size(32)
  .set_attr_direction(direction);

  std::vector<Operator> inputs{data_x};
  std::vector<Operator> outputs{common_lstm_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraphPtr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(computeGraphPtr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CommonLSTMFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraphPtr);

  bool findDynamicRNN = false;
  bool findStridedSlice = false;
  bool findConcat = false;
  for (auto node: computeGraphPtr->GetAllNodes()) {
    if (node->GetType() == "DynamicRNN") {
      findDynamicRNN = true;
    }
    if (node->GetType() == "StridedSliceD") {
      findStridedSlice = true;
    }
    if (node->GetType() == "ConcatD") {
      findConcat = true;
    }
  }
  EXPECT_EQ(findDynamicRNN, true);
  EXPECT_EQ(findStridedSlice, true);
  EXPECT_EQ(findConcat, true);
}

TEST_F(common_lstm_fusion_test, common_lstm_fusion_test_5) {
  ge::Graph graph("common_lstm_fusion_test_5");

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{1,2,160};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_w = op::Const("w");
  std::vector<int64_t> data_w_vec{2,32*4,160};
  ge::Shape data_w_shape(data_w_vec);
  ge::TensorDesc data_w_desc(data_w_shape, FORMAT_ND, DT_FLOAT);
  Tensor w_tensor;
  float * w_tensor_value = new float[160*32*4*2];
  w_tensor.SetTensorDesc(data_w_desc);
  w_tensor.SetData((uint8_t*)w_tensor_value, 160*32*4*2*sizeof(float));
  data_w.set_attr_value(w_tensor);
  data_w.update_output_desc_y(data_w_desc);

  // r
  auto data_r = op::Const("r");
  std::vector<int64_t> data_r_vec{2,32*4,32};
  ge::Shape data_r_shape(data_r_vec);
  ge::TensorDesc data_r_desc(data_r_shape, FORMAT_ND, DT_FLOAT);
  Tensor r_tensor;
  float * r_tensor_value = new float[32*32*4*2];
  r_tensor.SetTensorDesc(data_r_desc);
  r_tensor.SetData((uint8_t*)r_tensor_value, 32*32*4*2*sizeof(float));
  data_r.set_attr_value(r_tensor);
  data_r.update_output_desc_y(data_r_desc);
  // b
  auto data_b = op::Const("b");
  std::vector<int64_t> data_b_vec{2,32*4*2};
  ge::Shape data_b_shape(data_b_vec);
  ge::TensorDesc data_b_desc(data_b_shape, FORMAT_ND, DT_FLOAT);
  Tensor b_tensor;
  float * b_tensor_value = new float[2*32*4*2];
  b_tensor.SetTensorDesc(data_b_desc);
  b_tensor.SetData((uint8_t*)b_tensor_value, 2*32*4*2*sizeof(float));
  data_b.set_attr_value(b_tensor);
  data_b.update_output_desc_y(data_b_desc);
  //init_h
  auto data_h = op::Data("initial_h");
  std::vector<int64_t> data_h_vec{1,2,32};
  ge::Shape data_h_shape(data_h_vec);
  ge::TensorDesc data_h_desc(data_h_shape, FORMAT_ND, DT_FLOAT);
  data_h.update_input_desc_x(data_h_desc);
  data_h.update_output_desc_y(data_h_desc);
  //init_c
  auto data_c = op::Data("initial_c");
  std::vector<int64_t> data_c_vec{1,2,32};
  ge::Shape data_c_shape(data_c_vec);
  ge::TensorDesc data_c_desc(data_c_shape, FORMAT_ND, DT_FLOAT);
  data_c.update_input_desc_x(data_c_desc);
  data_c.update_output_desc_y(data_c_desc);

  auto common_lstm_op = op::CommonLSTM("CommonLSTM");
  string direction = "bidirectional";
  common_lstm_op.set_input_x(data_x)
  .set_input_w(data_w)
  .set_input_r(data_r)
  .set_input_b(data_b)
  .set_input_initial_h(data_h)
  .set_input_initial_c(data_c)
  .set_attr_hidden_size(32)
  .set_attr_direction(direction);

  std::vector<Operator> inputs{data_x, data_h, data_c};
  std::vector<Operator> outputs{common_lstm_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraphPtr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(computeGraphPtr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CommonLSTMFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraphPtr);

  bool findDynamicRNN = false;
  bool findStridedSlice = false;
  bool findConcat = false;
  bool findSplit = false;
  for (auto node: computeGraphPtr->GetAllNodes()) {
    if (node->GetType() == "DynamicRNN") {
      findDynamicRNN = true;
    }
    if (node->GetType() == "ConcatD") {
      findConcat = true;
    }
    if (node->GetType() == "SplitD") {
      findSplit = true;
    }
  }
  EXPECT_EQ(findDynamicRNN, true);
  EXPECT_EQ(findConcat, true);
  EXPECT_EQ(findSplit, true);
}

TEST_F(common_lstm_fusion_test, common_lstm_fusion_test_6) {
  // bidirectional lstm with const init_h, init_c
  ge::Graph graph("common_lstm_fusion_test_6");

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{1,2,160};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_w = op::Const("w");
  std::vector<int64_t> data_w_vec{2,32*4,160};
  ge::Shape data_w_shape(data_w_vec);
  ge::TensorDesc data_w_desc(data_w_shape, FORMAT_ND, DT_FLOAT);
  Tensor w_tensor;
  float * w_tensor_value = new float[160*32*4*2];
  w_tensor.SetTensorDesc(data_w_desc);
  w_tensor.SetData((uint8_t*)w_tensor_value, 160*32*4*2*sizeof(float));
  data_w.set_attr_value(w_tensor);
  data_w.update_output_desc_y(data_w_desc);

  // r
  auto data_r = op::Const("r");
  std::vector<int64_t> data_r_vec{2,32*4,32};
  ge::Shape data_r_shape(data_r_vec);
  ge::TensorDesc data_r_desc(data_r_shape, FORMAT_ND, DT_FLOAT);
  Tensor r_tensor;
  float * r_tensor_value = new float[32*32*4*2];
  r_tensor.SetTensorDesc(data_r_desc);
  r_tensor.SetData((uint8_t*)r_tensor_value, 32*32*4*2*sizeof(float));
  data_r.set_attr_value(r_tensor);
  data_r.update_output_desc_y(data_r_desc);
  // b
  auto data_b = op::Const("b");
  std::vector<int64_t> data_b_vec{2,32*4*2};
  ge::Shape data_b_shape(data_b_vec);
  ge::TensorDesc data_b_desc(data_b_shape, FORMAT_ND, DT_FLOAT);
  Tensor b_tensor;
  float * b_tensor_value = new float[2*32*4*2];
  b_tensor.SetTensorDesc(data_b_desc);
  b_tensor.SetData((uint8_t*)b_tensor_value, 2*32*4*2*sizeof(float));
  data_b.set_attr_value(b_tensor);
  data_b.update_output_desc_y(data_b_desc);
  //init_h
  auto data_h = op::Const("initial_h");
  std::vector<int64_t> data_h_vec{2,2,32};
  ge::Shape data_h_shape(data_h_vec);
  ge::TensorDesc data_h_desc(data_h_shape, FORMAT_ND, DT_FLOAT);
  Tensor h_tensor;
  float * h_tensor_value = new float[2*32*2];
  h_tensor.SetTensorDesc(data_h_desc);
  h_tensor.SetData((uint8_t*)h_tensor_value, 2*32*2*sizeof(float));
  data_h.set_attr_value(h_tensor);
  data_h.update_output_desc_y(data_h_desc);
  //init_c
  auto data_c = op::Const("initial_c");
  std::vector<int64_t> data_c_vec{2,2,32};
  ge::Shape data_c_shape(data_c_vec);
  ge::TensorDesc data_c_desc(data_c_shape, FORMAT_ND, DT_FLOAT);
  Tensor c_tensor;
  float * c_tensor_value = new float[2*32*2];
  c_tensor.SetTensorDesc(data_c_desc);
  c_tensor.SetData((uint8_t*)c_tensor_value, 2*32*2*sizeof(float));
  data_c.set_attr_value(c_tensor);
  data_c.update_output_desc_y(data_c_desc);

  auto common_lstm_op = op::CommonLSTM("CommonLSTM");
  string direction = "bidirectional";
  common_lstm_op.set_input_x(data_x)
  .set_input_w(data_w)
  .set_input_r(data_r)
  .set_input_b(data_b)
  .set_input_initial_h(data_h)
  .set_input_initial_c(data_c)
  .set_attr_hidden_size(32)
  .set_attr_direction(direction);

  std::vector<Operator> inputs{data_x};
  std::vector<Operator> outputs{common_lstm_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraphPtr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(computeGraphPtr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CommonLSTMFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraphPtr);

  bool findDynamicRNN = false;
  bool findStridedSlice = false;
  bool findConcat = false;
  bool findSplit = false;
  for (auto node: computeGraphPtr->GetAllNodes()) {
    if (node->GetType() == "DynamicRNN") {
      findDynamicRNN = true;
    }
    if (node->GetType() == "ConcatD") {
      findConcat = true;
    }
  }
  EXPECT_EQ(findDynamicRNN, true);
  EXPECT_EQ(findConcat, true);
}
