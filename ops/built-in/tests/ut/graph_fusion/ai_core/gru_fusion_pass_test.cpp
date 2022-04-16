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

class common_gru_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "common_gru_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "common_gru_fusion_test TearDown" << std::endl;
  }
};

TEST_F(common_gru_fusion_test, common_gru_fusion_test_1) {
  ge::Graph graph("common_gru_fusion_test_1");

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{2,2,10};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_w = op::Const("w");
  std::vector<int64_t> data_w_vec{1,48,10};
  ge::Shape data_w_shape(data_w_vec);
  ge::TensorDesc data_w_desc(data_w_shape, FORMAT_ND, DT_FLOAT);
  Tensor w_tensor;
  float * w_tensor_value = new float[10*48*1];
  w_tensor.SetTensorDesc(data_w_desc);
  w_tensor.SetData((uint8_t*)w_tensor_value, 10*48*1*sizeof(float));
  data_w.set_attr_value(w_tensor);
  // data_w.update_output_desc_y(data_w_desc);
  
  // r
  auto data_r = op::Const("r");
  std::vector<int64_t> data_r_vec{1,48,16};
  ge::Shape data_r_shape(data_r_vec);
  ge::TensorDesc data_r_desc(data_r_shape, FORMAT_ND, DT_FLOAT);
  Tensor r_tensor;
  float * r_tensor_value = new float[1,48,16];
  r_tensor.SetTensorDesc(data_r_desc);
  r_tensor.SetData((uint8_t*)r_tensor_value, 1*48*16*sizeof(float));
  data_r.set_attr_value(r_tensor);
  // data_r.update_output_desc_y(data_r_desc);

  // b
  auto data_b = op::Const("b");
  std::vector<int64_t> data_b_vec{1,96};
  ge::Shape data_b_shape(data_b_vec);
  ge::TensorDesc data_b_desc(data_b_shape, FORMAT_ND, DT_FLOAT);
  Tensor b_tensor;
  float * b_tensor_value = new float[1*96];
  b_tensor.SetTensorDesc(data_b_desc);
  b_tensor.SetData((uint8_t*)b_tensor_value, 1*96*sizeof(float));
  data_b.set_attr_value(b_tensor);
  // data_b.update_output_desc_y(data_b_desc);

  auto common_gru_op = op::CommonGRU("CommonGRU");
  common_gru_op.set_input_x(data_x)
           .set_input_w(data_w)
           .set_input_r(data_r)
           .set_input_b(data_b)
           .set_attr_hidden_size(16);
  std::vector<Operator> inputs{data_x, data_w, data_r, data_b};
  std::vector<Operator> outputs{common_gru_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraphPtr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(computeGraphPtr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CommonGRUFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraphPtr);

  bool findDynamicGRUV2 = false;
  bool findSplitD = false;

  for (auto node: computeGraphPtr->GetAllNodes()) {
    if (node->GetType() == "DynamicGRUV2" || node->GetType() == "DynamicGRUV2Hidden") {
       findDynamicGRUV2 = true;
     }

     if (node->GetType() == "SplitD") {
         findSplitD = true;
     }
  }
  EXPECT_EQ(findDynamicGRUV2, true);
  EXPECT_EQ(findSplitD, true);
}

TEST_F(common_gru_fusion_test, common_gru_fusion_test_2) {
  ge::Graph graph("common_gru_fusion_test_2");

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{2,2,10};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_w = op::Const("w");
  std::vector<int64_t> data_w_vec{2,48,10};
  ge::Shape data_w_shape(data_w_vec);
  ge::TensorDesc data_w_desc(data_w_shape, FORMAT_ND, DT_FLOAT);
  Tensor w_tensor;
  float * w_tensor_value = new float[10*48*2];
  w_tensor.SetTensorDesc(data_w_desc);
  w_tensor.SetData((uint8_t*)w_tensor_value, 10*48*2*sizeof(float));
  data_w.set_attr_value(w_tensor);
  // data_w.update_output_desc_y(data_w_desc);

  // r
  auto data_r = op::Const("r");
  std::vector<int64_t> data_r_vec{2,48,16};
  ge::Shape data_r_shape(data_r_vec);
  ge::TensorDesc data_r_desc(data_r_shape, FORMAT_ND, DT_FLOAT);
  Tensor r_tensor;
  float * r_tensor_value = new float[2,48,16];
  r_tensor.SetTensorDesc(data_r_desc);
  r_tensor.SetData((uint8_t*)r_tensor_value, 2*48*16*sizeof(float));
  data_r.set_attr_value(r_tensor);
  // data_r.update_output_desc_y(data_r_desc);

  // b
  auto data_b = op::Const("b");
  std::vector<int64_t> data_b_vec{2,96};
  ge::Shape data_b_shape(data_b_vec);
  ge::TensorDesc data_b_desc(data_b_shape, FORMAT_ND, DT_FLOAT);
  Tensor b_tensor;
  float * b_tensor_value = new float[2*96];
  b_tensor.SetTensorDesc(data_b_desc);
  b_tensor.SetData((uint8_t*)b_tensor_value, 2*96*sizeof(float));
  data_b.set_attr_value(b_tensor);
  // data_b.update_output_desc_y(data_b_desc);

  auto common_gru_op = op::CommonGRU("CommonGRU");
  common_gru_op.set_input_x(data_x)
           .set_input_w(data_w)
           .set_input_r(data_r)
           .set_input_b(data_b)
           .set_attr_hidden_size(16)
           .set_attr_direction("bidirectional");
  std::vector<Operator> inputs{data_x, data_w, data_r, data_b};
  std::vector<Operator> outputs{common_gru_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraphPtr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(computeGraphPtr);
  fe::FusionPassTestUtils::RunGraphFusionPass("CommonGRUFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraphPtr);

  bool findDynamicGRUV2 = false;

  for (auto node: computeGraphPtr->GetAllNodes()) {
    if (node->GetType() == "DynamicGRUV2" || node->GetType() == "DynamicGRUV2Hidden") {
       findDynamicGRUV2 = true;
     }
  }
  EXPECT_EQ(findDynamicGRUV2, true);
}