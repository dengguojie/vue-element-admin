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

class dynamic_gru_v2_transdatarnn_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dynamic_gru_v2_transdatarnn_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dynamic_gru_v2_transdatarnn_fusion_test TearDown" << std::endl;
  }
};


TEST_F(dynamic_gru_v2_transdatarnn_fusion_test, dynamic_gru_v2_transdatarnn_fusion_test_1) {
  ge::Graph graph("dynamic_gru_v2_transdatarnn_fusion_test_1");

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{1,1,15};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT16);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
 
  // w
  auto data_weight_input = op::Data("weight_input");
  std::vector<int64_t> data_weight_input_vec{15,45};
  ge::Shape data_weight_input_shape(data_weight_input_vec);
  ge::TensorDesc data_weight_input_desc(data_weight_input_shape, FORMAT_ND, DT_FLOAT16);
  data_weight_input.update_input_desc_x(data_weight_input_desc);
  data_weight_input.update_output_desc_y(data_weight_input_desc);

  auto data_weight_hidden = op::Data("weight_hidden");
  std::vector<int64_t> data_weight_hidden_vec{15,45};
  ge::Shape data_weight_hidden_shape(data_weight_hidden_vec);
  ge::TensorDesc data_weight_hidden_desc(data_weight_hidden_shape, FORMAT_ND, DT_FLOAT16);
  data_weight_hidden.update_input_desc_x(data_weight_hidden_desc);
  data_weight_hidden.update_output_desc_y(data_weight_hidden_desc);

  // bias_input
  auto data_bias_input = op::Data("bias_input");
  std::vector<int64_t> data_bias_input_vec{45};
  ge::Shape data_bias_input_shape(data_bias_input_vec);
  ge::TensorDesc data_bias_input_desc(data_bias_input_shape, FORMAT_ND, DT_FLOAT);
  data_bias_input.update_input_desc_x(data_bias_input_desc);
  data_bias_input.update_output_desc_y(data_bias_input_desc);

  // bias_hidden
  auto data_bias_hidden = op::Data("bias_hidden");
  std::vector<int64_t> data_bias_hidden_vec{45};
  ge::Shape data_bias_hidden_shape(data_bias_hidden_vec);
  ge::TensorDesc data_bias_hidden_desc(data_bias_hidden_shape, FORMAT_ND, DT_FLOAT);
  data_bias_hidden.update_input_desc_x(data_bias_hidden_desc);
  data_bias_hidden.update_output_desc_y(data_bias_hidden_desc);

  auto gru_v2_op = op::DynamicGRUV2("DynamicGRUV2");
  gru_v2_op.set_input_x(data_x)
           .set_input_weight_input(data_weight_input)
           .set_input_weight_hidden(data_weight_hidden)
           .set_input_bias_input(data_bias_input)
           .set_input_bias_hidden(data_bias_hidden);
           
  std::vector<Operator> inputs{data_x, data_weight_input, data_weight_hidden, data_bias_input, data_bias_hidden};
  std::vector<Operator> outputs{gru_v2_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicGRUV2TransFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

}

