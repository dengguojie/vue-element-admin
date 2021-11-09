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

class dynamic_rnn_grad_d_align_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dynamic_rnn_grad_d_align_fusion SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dynamic_rnn_grad_d_align_fusion TearDown" << std::endl;
  }
};

TEST_F(dynamic_rnn_grad_d_align_fusion_test, dynamic_rnn_grad_d_align_fusion_test_1) {
  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{-1, -1, 17};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT16);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_w = op::Data("w");
  std::vector<int64_t> data_w_vec{81, 256};
  ge::Shape data_w_shape(data_w_vec);
  ge::TensorDesc data_w_desc(data_w_shape, FORMAT_ND, DT_FLOAT16);
  data_w.update_input_desc_x(data_w_desc);
  data_w.update_output_desc_y(data_w_desc);

  // b
  auto data_b = op::Data("b");
  std::vector<int64_t> data_b_vec{256};
  ge::Shape data_b_shape(data_b_vec);
  ge::TensorDesc data_b_desc(data_b_shape, FORMAT_ND, DT_FLOAT16);
  data_b.update_input_desc_x(data_b_desc);
  data_b.update_output_desc_y(data_b_desc);

  // y
  auto data_y = op::Data("y");
  std::vector<int64_t> data_y_vec{-1, -1, 64};
  ge::Shape data_y_shape(data_y_vec);
  ge::TensorDesc data_y_desc(data_y_shape, FORMAT_ND, DT_FLOAT16);
  data_y.update_input_desc_x(data_y_desc);
  data_y.update_output_desc_y(data_y_desc);

  // init_h
  auto data_init_h = op::Data("init_h");
  std::vector<int64_t> data_init_h_vec{-1, 64};
  ge::Shape data_init_h_shape(data_init_h_vec);
  ge::TensorDesc data_init_h_desc(data_init_h_shape, FORMAT_ND, DT_FLOAT16);
  data_init_h.update_input_desc_x(data_init_h_desc);
  data_init_h.update_output_desc_y(data_init_h_desc);

  // init_c
  auto data_init_c = op::Data("init_c");
  std::vector<int64_t> data_init_c_vec{-1, 64};
  ge::Shape data_init_c_shape(data_init_c_vec);
  ge::TensorDesc data_init_c_desc(data_init_c_shape, FORMAT_ND, DT_FLOAT16);
  data_init_c.update_input_desc_x(data_init_c_desc);
  data_init_c.update_output_desc_y(data_init_c_desc);

  // h
  auto data_h = op::Data("h");
  std::vector<int64_t> data_h_vec{-1, -1, 64};
  ge::Shape data_h_shape(data_h_vec);
  ge::TensorDesc data_h_desc(data_h_shape, FORMAT_ND, DT_FLOAT16);
  data_h.update_input_desc_x(data_h_desc);
  data_h.update_output_desc_y(data_h_desc);

  // c
  auto data_c = op::Data("c");
  std::vector<int64_t> data_c_vec{-1, -1, 64};
  ge::Shape data_c_shape(data_c_vec);
  ge::TensorDesc data_c_desc(data_c_shape, FORMAT_ND, DT_FLOAT16);
  data_c.update_input_desc_x(data_c_desc);
  data_c.update_output_desc_y(data_c_desc);

  // dy
  auto data_dy = op::Data("dy");
  std::vector<int64_t> data_dy_vec{-1, -1, 64};
  ge::Shape data_dy_shape(data_dy_vec);
  ge::TensorDesc data_dy_desc(data_dy_shape, FORMAT_ND, DT_FLOAT16);
  data_dy.update_input_desc_x(data_dy_desc);
  data_dy.update_output_desc_y(data_dy_desc);

  // dh
  auto data_dh = op::Data("dh");
  std::vector<int64_t> data_dh_vec{-1, 64};
  ge::Shape data_dh_shape(data_dh_vec);
  ge::TensorDesc data_dh_desc(data_dh_shape, FORMAT_ND, DT_FLOAT16);
  data_dh.update_input_desc_x(data_dh_desc);
  data_dh.update_output_desc_y(data_dh_desc);

  // dc
  auto data_dc = op::Data("dc");
  std::vector<int64_t> data_dc_vec{-1, 64};
  ge::Shape data_dc_shape(data_dc_vec);
  ge::TensorDesc data_dc_desc(data_dc_shape, FORMAT_ND, DT_FLOAT16);
  data_dc.update_input_desc_x(data_dc_desc);
  data_dc.update_output_desc_y(data_dc_desc);

  // i
  auto data_i = op::Data("i");
  std::vector<int64_t> data_i_vec{-1, -1, 64};
  ge::Shape data_i_shape(data_i_vec);
  ge::TensorDesc data_i_desc(data_i_shape, FORMAT_ND, DT_FLOAT16);
  data_i.update_input_desc_x(data_i_desc);
  data_i.update_output_desc_y(data_i_desc);

  // j
  auto data_j = op::Data("j");
  std::vector<int64_t> data_j_vec{-1, -1, 64};
  ge::Shape data_j_shape(data_j_vec);
  ge::TensorDesc data_j_desc(data_j_shape, FORMAT_ND, DT_FLOAT16);
  data_j.update_input_desc_x(data_j_desc);
  data_j.update_output_desc_y(data_j_desc);

  // f
  auto data_f = op::Data("f");
  std::vector<int64_t> data_f_vec{-1, -1, 64};
  ge::Shape data_f_shape(data_f_vec);
  ge::TensorDesc data_f_desc(data_f_shape, FORMAT_ND, DT_FLOAT16);
  data_f.update_input_desc_x(data_f_desc);
  data_f.update_output_desc_y(data_f_desc);

  // o
  auto data_o = op::Data("o");
  std::vector<int64_t> data_o_vec{-1, -1, 64};
  ge::Shape data_o_shape(data_o_vec);
  ge::TensorDesc data_o_desc(data_o_shape, FORMAT_ND, DT_FLOAT16);
  data_o.update_input_desc_x(data_o_desc);
  data_o.update_output_desc_y(data_o_desc);

  // tanhct
  auto data_tanhct = op::Data("tanhct");
  std::vector<int64_t> data_tanhct_vec{-1, -1, 64};
  ge::Shape data_tanhct_shape(data_tanhct_vec);
  ge::TensorDesc data_tanhct_desc(data_tanhct_shape, FORMAT_ND, DT_FLOAT16);
  data_tanhct.update_input_desc_x(data_tanhct_desc);
  data_tanhct.update_output_desc_y(data_tanhct_desc);

  // tanhct
  auto data_seqlength = op::Data("seq_length");
  std::vector<int64_t> data_seqlength_vec{-1, -1, 64};
  ge::Shape data_seqlength_shape(data_seqlength_vec);
  ge::TensorDesc data_seqlength_desc(data_seqlength_shape, FORMAT_ND, DT_FLOAT16);
  data_seqlength.update_input_desc_x(data_seqlength_desc);
  data_seqlength.update_output_desc_y(data_seqlength_desc);

  auto dynamci_rnn_grad_op = op::DynamicRNNGrad("DynamicRNNGrad");
  dynamci_rnn_grad_op.set_input_x(data_x)
      .set_input_w(data_w)
      .set_input_b(data_b)
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
      .set_input_tanhct(data_tanhct)
      .set_input_seq_length(data_seqlength);
  std::vector<Operator> inputs{data_x, data_w, data_b,  data_y,      data_init_h,   data_init_c,
                               data_h, data_c, data_dy, data_dh,     data_dc,       data_i,
                               data_j, data_f, data_o,  data_tanhct, data_seqlength};

  std::vector<Operator> outputs{dynamci_rnn_grad_op};
  ge::Graph graph("dynamic_rnn_grad_d_align_fusion_test_1");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  std::string session_id = "testGraph";
  auto res2 = AttrUtils::SetStr(compute_graph_ptr, "_session_graph_id", session_id);

  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicRNNGradDAlignFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findMatMulV2 = false;
  bool findSplit = false;
  bool findConcat = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMulV2") {
      findMatMulV2 = true;
    }

    if (node->GetType() == "SplitVD") {
      findSplit = true;
    }

    if (node->GetType() == "ConcatD") {
      findConcat = true;
    }
  }

  EXPECT_EQ(findMatMulV2, true);
  //  EXPECT_EQ(findSplit, true);
  //  EXPECT_EQ(findConcat, true);
}

TEST_F(dynamic_rnn_grad_d_align_fusion_test, dynamic_rnn_grad_d_align_fusion_test_2) {
  // x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{-1, -1, 32};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT16);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  // w
  auto data_w = op::Data("w");
  std::vector<int64_t> data_w_vec{82, 200};
  ge::Shape data_w_shape(data_w_vec);
  ge::TensorDesc data_w_desc(data_w_shape, FORMAT_ND, DT_FLOAT16);
  data_w.update_input_desc_x(data_w_desc);
  data_w.update_output_desc_y(data_w_desc);

  // b
  auto data_b = op::Data("b");
  std::vector<int64_t> data_b_vec{200};
  ge::Shape data_b_shape(data_b_vec);
  ge::TensorDesc data_b_desc(data_b_shape, FORMAT_ND, DT_FLOAT16);
  data_b.update_input_desc_x(data_b_desc);
  data_b.update_output_desc_y(data_b_desc);

  // y
  auto data_y = op::Data("y");
  std::vector<int64_t> data_y_vec{-1, -1, 50};
  ge::Shape data_y_shape(data_y_vec);
  ge::TensorDesc data_y_desc(data_y_shape, FORMAT_ND, DT_FLOAT16);
  data_y.update_input_desc_x(data_y_desc);
  data_y.update_output_desc_y(data_y_desc);

  // init_h
  auto data_init_h = op::Data("init_h");
  std::vector<int64_t> data_init_h_vec{-1, 50};
  ge::Shape data_init_h_shape(data_init_h_vec);
  ge::TensorDesc data_init_h_desc(data_init_h_shape, FORMAT_ND, DT_FLOAT16);
  data_init_h.update_input_desc_x(data_init_h_desc);
  data_init_h.update_output_desc_y(data_init_h_desc);

  // init_c
  auto data_init_c = op::Data("init_c");
  std::vector<int64_t> data_init_c_vec{1, -1, 50};
  ge::Shape data_init_c_shape(data_init_c_vec);
  ge::TensorDesc data_init_c_desc(data_init_c_shape, FORMAT_ND, DT_FLOAT16);
  data_init_c.update_input_desc_x(data_init_c_desc);
  data_init_c.update_output_desc_y(data_init_c_desc);

  // h
  auto data_h = op::Data("h");
  std::vector<int64_t> data_h_vec{-1, -1, 50};
  ge::Shape data_h_shape(data_h_vec);
  ge::TensorDesc data_h_desc(data_h_shape, FORMAT_ND, DT_FLOAT16);
  data_h.update_input_desc_x(data_h_desc);
  data_h.update_output_desc_y(data_h_desc);

  // c
  auto data_c = op::Data("c");
  std::vector<int64_t> data_c_vec{-1, -1, 50};
  ge::Shape data_c_shape(data_c_vec);
  ge::TensorDesc data_c_desc(data_c_shape, FORMAT_ND, DT_FLOAT16);
  data_c.update_input_desc_x(data_c_desc);
  data_c.update_output_desc_y(data_c_desc);

  // dy
  auto data_dy = op::Data("dy");
  std::vector<int64_t> data_dy_vec{-1, -1, 50};
  ge::Shape data_dy_shape(data_dy_vec);
  ge::TensorDesc data_dy_desc(data_dy_shape, FORMAT_ND, DT_FLOAT16);
  data_dy.update_input_desc_x(data_dy_desc);
  data_dy.update_output_desc_y(data_dy_desc);

  // dh
  auto data_dh = op::Data("dh");
  std::vector<int64_t> data_dh_vec{1, -1, 50};
  ge::Shape data_dh_shape(data_dh_vec);
  ge::TensorDesc data_dh_desc(data_dh_shape, FORMAT_ND, DT_FLOAT16);
  data_dh.update_input_desc_x(data_dh_desc);
  data_dh.update_output_desc_y(data_dh_desc);

  // dc
  auto data_dc = op::Data("dc");
  std::vector<int64_t> data_dc_vec{1, -1, 50};
  ge::Shape data_dc_shape(data_dc_vec);
  ge::TensorDesc data_dc_desc(data_dc_shape, FORMAT_ND, DT_FLOAT16);
  data_dc.update_input_desc_x(data_dc_desc);
  data_dc.update_output_desc_y(data_dc_desc);

  // i
  auto data_i = op::Data("i");
  std::vector<int64_t> data_i_vec{-1, -1, 50};
  ge::Shape data_i_shape(data_i_vec);
  ge::TensorDesc data_i_desc(data_i_shape, FORMAT_ND, DT_FLOAT16);
  data_i.update_input_desc_x(data_i_desc);
  data_i.update_output_desc_y(data_i_desc);

  // j
  auto data_j = op::Data("j");
  std::vector<int64_t> data_j_vec{-1, -1, 50};
  ge::Shape data_j_shape(data_j_vec);
  ge::TensorDesc data_j_desc(data_j_shape, FORMAT_ND, DT_FLOAT16);
  data_j.update_input_desc_x(data_j_desc);
  data_j.update_output_desc_y(data_j_desc);

  // f
  auto data_f = op::Data("f");
  std::vector<int64_t> data_f_vec{-1, -1, 50};
  ge::Shape data_f_shape(data_f_vec);
  ge::TensorDesc data_f_desc(data_f_shape, FORMAT_ND, DT_FLOAT16);
  data_f.update_input_desc_x(data_f_desc);
  data_f.update_output_desc_y(data_f_desc);

  // o
  auto data_o = op::Data("o");
  std::vector<int64_t> data_o_vec{-1, -1, 50};
  ge::Shape data_o_shape(data_o_vec);
  ge::TensorDesc data_o_desc(data_o_shape, FORMAT_ND, DT_FLOAT16);
  data_o.update_input_desc_x(data_o_desc);
  data_o.update_output_desc_y(data_o_desc);

  // tanhct
  auto data_tanhct = op::Data("tanhct");
  std::vector<int64_t> data_tanhct_vec{-1, -1, 50};
  ge::Shape data_tanhct_shape(data_tanhct_vec);
  ge::TensorDesc data_tanhct_desc(data_tanhct_shape, FORMAT_ND, DT_FLOAT16);
  data_tanhct.update_input_desc_x(data_tanhct_desc);
  data_tanhct.update_output_desc_y(data_tanhct_desc);

  // tanhct
  auto data_seqlength = op::Data("seq_length");
  std::vector<int64_t> data_seqlength_vec{-1, -1, 50};
  ge::Shape data_seqlength_shape(data_seqlength_vec);
  ge::TensorDesc data_seqlength_desc(data_seqlength_shape, FORMAT_ND, DT_FLOAT16);
  data_seqlength.update_input_desc_x(data_seqlength_desc);
  data_seqlength.update_output_desc_y(data_seqlength_desc);

  auto dynamci_rnn_grad_op = op::DynamicRNNGrad("DynamicRNNGrad");
  dynamci_rnn_grad_op.set_input_x(data_x)
      .set_input_w(data_w)
      .set_input_b(data_b)
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
      .set_input_tanhct(data_tanhct)
      .set_input_seq_length(data_seqlength);
  std::vector<Operator> inputs{data_x, data_w, data_b,  data_y,      data_init_h,   data_init_c,
                               data_h, data_c, data_dy, data_dh,     data_dc,       data_i,
                               data_j, data_f, data_o,  data_tanhct, data_seqlength};

  std::vector<Operator> outputs{dynamci_rnn_grad_op};
  ge::Graph graph("dynamic_rnn_grad_d_align_fusion_test_2");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  std::string session_id = "testGraph";
  auto res2 = AttrUtils::SetStr(compute_graph_ptr, "_session_graph_id", session_id);

  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicRNNGradDAlignFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findMatMulV2 = false;
  bool findSplit = false;
  bool findConcat = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMulV2") {
      findMatMulV2 = true;
    }

    if (node->GetType() == "SplitVD") {
      findSplit = true;
    }

    if (node->GetType() == "ConcatD") {
      findConcat = true;
    }
  }

  EXPECT_EQ(findMatMulV2, true);
}