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

class dynamic_rnn_v2_grad_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_rnn_v2_grad_fusion SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dynamic_rnn_v2_grad_fusion TearDown" << std::endl;
    }
};

TEST_F(dynamic_rnn_v2_grad_fusion_test, dynamic_rnn_v2_grad_fusion_test_1) {
  int time_step = 1;
  int batch_size = -1;
  int input_size = 16;
  int hidden_size = 16;
  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{batch_size, input_size};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT16);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // wx
  auto data_wx = op::Data("wx");
  std::vector<int64_t> data_wx_vec{input_size, 4 * hidden_size};
  ge::Shape data_wx_shape(data_wx_vec);
  ge::TensorDesc data_wx_desc(data_wx_shape, FORMAT_ND, DT_FLOAT16);
  data_wx.update_input_desc_x(data_wx_desc);
  data_wx.update_output_desc_y(data_wx_desc);

  // wh
  auto data_wh = op::Data("wh");
  std::vector<int64_t> data_wh_vec{input_size, 4 * hidden_size};
  ge::Shape data_wh_shape(data_wh_vec);
  ge::TensorDesc data_wh_desc(data_wh_shape, FORMAT_ND, DT_FLOAT16);
  data_wh.update_input_desc_x(data_wh_desc);
  data_wh.update_output_desc_y(data_wh_desc);

  std::vector<int64_t> data_2d_shape{batch_size, hidden_size};
  std::vector<int64_t> data_3d_shape{time_step, batch_size, hidden_size};
  // y
  auto data_y = op::Data("y");
  ge::Shape data_y_shape(data_3d_shape);
  ge::TensorDesc data_y_desc(data_y_shape, FORMAT_ND, DT_FLOAT16);
  data_y.update_input_desc_x(data_y_desc);
  data_y.update_output_desc_y(data_y_desc);

  // init_h
  auto data_init_h = op::Data("init_h");
  ge::Shape data_init_h_shape(data_2d_shape);
  ge::TensorDesc data_init_h_desc(data_init_h_shape, FORMAT_ND, DT_FLOAT16);
  data_init_h.update_input_desc_x(data_init_h_desc);
  data_init_h.update_output_desc_y(data_init_h_desc);

  // init_c
  auto data_init_c = op::Data("init_c");
  ge::Shape data_init_c_shape(data_2d_shape);
  ge::TensorDesc data_init_c_desc(data_init_c_shape, FORMAT_ND, DT_FLOAT16);
  data_init_c.update_input_desc_x(data_init_c_desc);
  data_init_c.update_output_desc_y(data_init_c_desc);

  // h
  auto data_h = op::Data("h");
  ge::Shape data_h_shape(data_3d_shape);
  ge::TensorDesc data_h_desc(data_h_shape, FORMAT_ND, DT_FLOAT16);
  data_h.update_input_desc_x(data_h_desc);
  data_h.update_output_desc_y(data_h_desc);

  // c
  auto data_c = op::Data("c");
  ge::Shape data_c_shape(data_3d_shape);
  ge::TensorDesc data_c_desc(data_c_shape, FORMAT_ND, DT_FLOAT16);
  data_c.update_input_desc_x(data_c_desc);
  data_c.update_output_desc_y(data_c_desc);

  std::vector<int64_t> data_gate_vec{time_step, batch_size, hidden_size};
  // dy
  auto data_dy = op::Data("dy");
  ge::Shape data_dy_shape(data_3d_shape);
  ge::TensorDesc data_dy_desc(data_dy_shape, FORMAT_ND, DT_FLOAT16);
  data_dy.update_input_desc_x(data_dy_desc);
  data_dy.update_output_desc_y(data_dy_desc);

  // dh
  auto data_dh = op::Data("dh");
  ge::Shape data_dh_shape(data_2d_shape);
  ge::TensorDesc data_dh_desc(data_dh_shape, FORMAT_ND, DT_FLOAT16);
  data_dh.update_input_desc_x(data_dh_desc);
  data_dh.update_output_desc_y(data_dh_desc);

  // dc
  auto data_dc = op::Data("dc");
  ge::Shape data_dc_shape(data_2d_shape);
  ge::TensorDesc data_dc_desc(data_dc_shape, FORMAT_ND, DT_FLOAT16);
  data_dc.update_input_desc_x(data_dc_desc);
  data_dc.update_output_desc_y(data_dc_desc);

  // i
  auto data_i = op::Data("i");
  ge::Shape data_i_shape(data_3d_shape);
  ge::TensorDesc data_i_desc(data_i_shape, FORMAT_ND, DT_FLOAT16);
  data_i.update_input_desc_x(data_i_desc);
  data_i.update_output_desc_y(data_i_desc);

  // j
  auto data_j = op::Data("j");
  ge::Shape data_j_shape(data_3d_shape);
  ge::TensorDesc data_j_desc(data_j_shape, FORMAT_ND, DT_FLOAT16);
  data_j.update_input_desc_x(data_j_desc);
  data_j.update_output_desc_y(data_j_desc);

  // f
  auto data_f = op::Data("f");
  ge::Shape data_f_shape(data_3d_shape);
  ge::TensorDesc data_f_desc(data_f_shape, FORMAT_ND, DT_FLOAT16);
  data_f.update_input_desc_x(data_f_desc);
  data_f.update_output_desc_y(data_f_desc);

  // o
  auto data_o = op::Data("o");
  ge::Shape data_o_shape(data_3d_shape);
  ge::TensorDesc data_o_desc(data_o_shape, FORMAT_ND, DT_FLOAT16);
  data_o.update_input_desc_x(data_o_desc);
  data_o.update_output_desc_y(data_o_desc);

  // tanhct
  auto data_tanhct = op::Data("tanhct");
  ge::Shape data_tanhct_shape(data_3d_shape);
  ge::TensorDesc data_tanhct_desc(data_tanhct_shape, FORMAT_ND, DT_FLOAT16);
  data_tanhct.update_input_desc_x(data_tanhct_desc);
  data_tanhct.update_output_desc_y(data_tanhct_desc);

  auto dynamci_rnn_grad_op = op::DynamicRNNV2Grad("DynamicRNNV2Grad");
  dynamci_rnn_grad_op.set_input_x(data_x)
      .set_input_w_x(data_wx)
      .set_input_w_h(data_wh)
      .set_input_y(data_y)
      .set_input_init_h(data_init_h)
      .set_input_init_c(data_init_c)
      .set_input_h(data_h)
      .set_input_c(data_c)
      .set_input_dy(data_dy)
      .set_input_dh(data_dh)
      .set_input_dc(data_dc)
      .set_input_i(data_i)
      .set_input_j(data_j)
      .set_input_f(data_f)
      .set_input_o(data_o)
      .set_input_tanhct(data_tanhct);
  std::vector<Operator> inputs{data_x,  data_wx, data_wh, data_y, data_init_h, data_init_c, data_h, data_c,
                               data_dy, data_dh, data_dc, data_i, data_j,      data_f,      data_o, data_tanhct};

  std::vector<Operator> outputs{dynamci_rnn_grad_op};
  ge::Graph graph("dynamic_rnn_v2_grad_fusion_test_1");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  std::string session_id = "testGraph";
  auto res2 = AttrUtils::SetStr(compute_graph_ptr, "_session_graph_id", session_id);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicRNNV2GradFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findDynamicLSTMGradCell = false;
  bool findBatchMatMulV2 = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "DynamicLSTMGradCell") {
      findDynamicLSTMGradCell = true;
    } else if (node->GetType() == "BatchMatMulV2") {
      findBatchMatMulV2 = true;
    }
  }
  EXPECT_EQ(findDynamicLSTMGradCell, true);
  EXPECT_EQ(findBatchMatMulV2, true);
}
