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

class dynamic_gru_v2_grad_align_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_gru_v2_grad_align_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dynamic_gru_v2_grad_align_fusion_test TearDown" << std::endl;
    }
};

TEST_F(dynamic_gru_v2_grad_align_fusion_test, dynamic_gru_v2_grad_align_fusion_test_1) {

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{1,1,17};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT16);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // weight_input
  auto data_w = op::Data("weight_input");
  std::vector<int64_t> data_w_vec{17,51};
  ge::Shape data_w_shape(data_w_vec);
  ge::TensorDesc data_w_desc(data_w_shape, FORMAT_ND, DT_FLOAT16);
  data_w.update_input_desc_x(data_w_desc);
  data_w.update_output_desc_y(data_w_desc);
  
  // b
  auto data_b = op::Data("weight_hidden");
  std::vector<int64_t> data_b_vec{17,51};
  ge::Shape data_b_shape(data_b_vec);
  ge::TensorDesc data_b_desc(data_b_shape, FORMAT_ND, DT_FLOAT16);
  data_b.update_input_desc_x(data_b_desc);
  data_b.update_output_desc_y(data_b_desc);

  // y
  auto data_y = op::Data("y");
  std::vector<int64_t> data_y_vec{1,1,17};
  ge::Shape data_y_shape(data_y_vec);
  ge::TensorDesc data_y_desc(data_y_shape, FORMAT_ND, DT_FLOAT16);
  data_y.update_input_desc_x(data_y_desc);
  data_y.update_output_desc_y(data_y_desc);

  // init_h
  auto data_init_h = op::Data("init_h");
  std::vector<int64_t> data_init_h_vec{1,1,17};
  ge::Shape data_init_h_shape(data_init_h_vec);
  ge::TensorDesc data_init_h_desc(data_init_h_shape, FORMAT_ND, DT_FLOAT16);
  data_init_h.update_input_desc_x(data_init_h_desc);
  data_init_h.update_output_desc_y(data_init_h_desc);

  // init_c
  auto data_init_c = op::Data("h");
  std::vector<int64_t> data_init_c_vec{1,1,17};
  ge::Shape data_init_c_shape(data_init_c_vec);
  ge::TensorDesc data_init_c_desc(data_init_c_shape, FORMAT_ND, DT_FLOAT16);
  data_init_c.update_input_desc_x(data_init_c_desc);
  data_init_c.update_output_desc_y(data_init_c_desc);

  // dy
  auto data_dy = op::Data("dy");
  std::vector<int64_t> data_dy_vec{1,1,17};
  ge::Shape data_dy_shape(data_dy_vec);
  ge::TensorDesc data_dy_desc(data_dy_shape, FORMAT_ND, DT_FLOAT16);
  data_dy.update_input_desc_x(data_dy_desc);
  data_dy.update_output_desc_y(data_dy_desc);

  // dh
  auto data_dh = op::Data("dh");
  std::vector<int64_t> data_dh_vec{1,1,17};
  ge::Shape data_dh_shape(data_dh_vec);
  ge::TensorDesc data_dh_desc(data_dh_shape, FORMAT_ND, DT_FLOAT16);
  data_dh.update_input_desc_x(data_dh_desc);
  data_dh.update_output_desc_y(data_dh_desc);

  // i
  auto data_i = op::Data("update");
  std::vector<int64_t> data_i_vec{1,1,17};
  ge::Shape data_i_shape(data_i_vec);
  ge::TensorDesc data_i_desc(data_i_shape, FORMAT_ND, DT_FLOAT16);
  data_i.update_input_desc_x(data_i_desc);
  data_i.update_output_desc_y(data_i_desc);

  // j
  auto data_j = op::Data("reset");
  std::vector<int64_t> data_j_vec{1,1,17};
  ge::Shape data_j_shape(data_j_vec);
  ge::TensorDesc data_j_desc(data_j_shape, FORMAT_ND, DT_FLOAT16);
  data_j.update_input_desc_x(data_j_desc);
  data_j.update_output_desc_y(data_j_desc);

  // f
  auto data_f = op::Data("new");
  std::vector<int64_t> data_f_vec{1,1,17};
  ge::Shape data_f_shape(data_f_vec);
  ge::TensorDesc data_f_desc(data_f_shape, FORMAT_ND, DT_FLOAT16);
  data_f.update_input_desc_x(data_f_desc);
  data_f.update_output_desc_y(data_f_desc);

  // o
  auto data_o = op::Data("hidden_new");
  std::vector<int64_t> data_o_vec{1,1,17};
  ge::Shape data_o_shape(data_o_vec);
  ge::TensorDesc data_o_desc(data_o_shape, FORMAT_ND, DT_FLOAT16);
  data_o.update_input_desc_x(data_o_desc);
  data_o.update_output_desc_y(data_o_desc);
  
  auto dynamci_gru_v2_grad_op = op::DynamicGRUV2Grad("DynamicGRUV2Grad");
  dynamci_gru_v2_grad_op.set_input_x(data_x)
                        .set_input_weight_input(data_w)
                        .set_input_weight_hidden(data_b)
                        .set_input_y(data_y)
                        .set_input_init_h(data_init_h)
                        .set_input_h(data_init_c)
                        .set_input_dy(data_dy)
                        .set_input_dh(data_dh)
                        .set_input_update(data_i)
                        .set_input_reset(data_j)
                        .set_input_new(data_f)
                        .set_input_hidden_new(data_o);

  std::vector<Operator> inputs{data_x, data_w, data_b, data_y, data_init_h, data_init_c, 
                               data_dy, data_dh, data_i, data_j, data_f, data_o};
  std::vector<Operator> outputs{dynamci_gru_v2_grad_op};

  ge::Graph graph("dynamic_gru_v2_grad_align_fusion_test");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicGRUV2GradAlignFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

}

TEST_F(dynamic_gru_v2_grad_align_fusion_test, dynamic_gru_v2_grad_align_fusion_test_2) {

  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{2,1,17};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT16);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // weight_input
  auto data_w = op::Data("weight_input");
  std::vector<int64_t> data_w_vec{17,51};
  ge::Shape data_w_shape(data_w_vec);
  ge::TensorDesc data_w_desc(data_w_shape, FORMAT_ND, DT_FLOAT16);
  data_w.update_input_desc_x(data_w_desc);
  data_w.update_output_desc_y(data_w_desc);
  
  // b
  auto data_b = op::Data("weight_hidden");
  std::vector<int64_t> data_b_vec{17,51};
  ge::Shape data_b_shape(data_b_vec);
  ge::TensorDesc data_b_desc(data_b_shape, FORMAT_ND, DT_FLOAT16);
  data_b.update_input_desc_x(data_b_desc);
  data_b.update_output_desc_y(data_b_desc);

  // y
  auto data_y = op::Data("y");
  std::vector<int64_t> data_y_vec{2,1,17};
  ge::Shape data_y_shape(data_y_vec);
  ge::TensorDesc data_y_desc(data_y_shape, FORMAT_ND, DT_FLOAT16);
  data_y.update_input_desc_x(data_y_desc);
  data_y.update_output_desc_y(data_y_desc);

  // init_h
  auto data_init_h = op::Data("init_h");
  std::vector<int64_t> data_init_h_vec{1,1,17};
  ge::Shape data_init_h_shape(data_init_h_vec);
  ge::TensorDesc data_init_h_desc(data_init_h_shape, FORMAT_ND, DT_FLOAT16);
  data_init_h.update_input_desc_x(data_init_h_desc);
  data_init_h.update_output_desc_y(data_init_h_desc);

  // init_c
  auto data_init_c = op::Data("h");
  std::vector<int64_t> data_init_c_vec{2,1,17};
  ge::Shape data_init_c_shape(data_init_c_vec);
  ge::TensorDesc data_init_c_desc(data_init_c_shape, FORMAT_ND, DT_FLOAT16);
  data_init_c.update_input_desc_x(data_init_c_desc);
  data_init_c.update_output_desc_y(data_init_c_desc);

  // dy
  auto data_dy = op::Data("dy");
  std::vector<int64_t> data_dy_vec{2,1,17};
  ge::Shape data_dy_shape(data_dy_vec);
  ge::TensorDesc data_dy_desc(data_dy_shape, FORMAT_ND, DT_FLOAT16);
  data_dy.update_input_desc_x(data_dy_desc);
  data_dy.update_output_desc_y(data_dy_desc);

  // dh
  auto data_dh = op::Data("dh");
  std::vector<int64_t> data_dh_vec{1,1,17};
  ge::Shape data_dh_shape(data_dh_vec);
  ge::TensorDesc data_dh_desc(data_dh_shape, FORMAT_ND, DT_FLOAT16);
  data_dh.update_input_desc_x(data_dh_desc);
  data_dh.update_output_desc_y(data_dh_desc);

  // i
  auto data_i = op::Data("update");
  std::vector<int64_t> data_i_vec{2,1,17};
  ge::Shape data_i_shape(data_i_vec);
  ge::TensorDesc data_i_desc(data_i_shape, FORMAT_ND, DT_FLOAT16);
  data_i.update_input_desc_x(data_i_desc);
  data_i.update_output_desc_y(data_i_desc);

  // j
  auto data_j = op::Data("reset");
  std::vector<int64_t> data_j_vec{2,1,17};
  ge::Shape data_j_shape(data_j_vec);
  ge::TensorDesc data_j_desc(data_j_shape, FORMAT_ND, DT_FLOAT16);
  data_j.update_input_desc_x(data_j_desc);
  data_j.update_output_desc_y(data_j_desc);

  // f
  auto data_f = op::Data("new");
  std::vector<int64_t> data_f_vec{2,1,17};
  ge::Shape data_f_shape(data_f_vec);
  ge::TensorDesc data_f_desc(data_f_shape, FORMAT_ND, DT_FLOAT16);
  data_f.update_input_desc_x(data_f_desc);
  data_f.update_output_desc_y(data_f_desc);

  // o
  auto data_o = op::Data("hidden_new");
  std::vector<int64_t> data_o_vec{2,1,17};
  ge::Shape data_o_shape(data_o_vec);
  ge::TensorDesc data_o_desc(data_o_shape, FORMAT_ND, DT_FLOAT16);
  data_o.update_input_desc_x(data_o_desc);
  data_o.update_output_desc_y(data_o_desc);
  
  auto dynamci_gru_v2_grad_op = op::DynamicGRUV2Grad("DynamicGRUV2Grad");
  dynamci_gru_v2_grad_op.set_input_x(data_x)
                        .set_input_weight_input(data_w)
                        .set_input_weight_hidden(data_b)
                        .set_input_y(data_y)
                        .set_input_init_h(data_init_h)
                        .set_input_h(data_init_c)
                        .set_input_dy(data_dy)
                        .set_input_dh(data_dh)
                        .set_input_update(data_i)
                        .set_input_reset(data_j)
                        .set_input_new(data_f)
                        .set_input_hidden_new(data_o);

  std::vector<Operator> inputs{data_x, data_w, data_b, data_y, data_init_h, data_init_c, 
                               data_dy, data_dh, data_i, data_j, data_f, data_o};
  std::vector<Operator> outputs{dynamci_gru_v2_grad_op};

  ge::Graph graph("dynamic_gru_v2_grad_align_fusion_test");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicGRUV2GradAlignFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

}
