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

class rnn_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "rnn_fusion SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "rnn_fusion TearDown" << std::endl;
    }
};

TEST_F(rnn_fusion_test, rnn_fusion_test_1) {

  //x
  auto data_x = op::Data("x");
  std::vector<int64_t> data_x_vec{10,18,1024};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT16);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);

  //w
  auto data_cont = op::Data("cont");
  std::vector<int64_t> data_cont_vec{10, 18};
  ge::Shape data_cont_shape(data_cont_vec);
  ge::TensorDesc data_cont_desc(data_cont_shape, FORMAT_ND, DT_FLOAT16);
  data_cont.update_input_desc_x(data_cont_desc);
  data_cont.update_output_desc_y(data_cont_desc);
  
  //b
  auto data_x_static = op::Data("x_static");
  std::vector<int64_t> data_x_static_vec{18, 18};
  ge::Shape data_x_static_shape(data_x_static_vec);
  ge::TensorDesc data_x_static_desc(data_x_static_shape, FORMAT_ND, DT_FLOAT16);
  data_x_static.update_input_desc_x(data_x_static_desc);
  data_x_static.update_output_desc_y(data_x_static_desc);

  //y
  auto data_h_0 = op::Data("h_0");
  std::vector<int64_t> data_h_0_vec{1,18,512};
  ge::Shape data_h_0_shape(data_h_0_vec);
  ge::TensorDesc data_h_0_desc(data_h_0_shape, FORMAT_ND, DT_FLOAT16);
  data_h_0.update_input_desc_x(data_h_0_desc);
  data_h_0.update_output_desc_y(data_h_0_desc);

  //init_h
  auto data_w_xh = op::Data("w_xh");
  std::vector<int64_t> data_w_xh_vec{512, 1024};
  ge::Shape data_w_xh_shape(data_w_xh_vec);
  ge::TensorDesc data_w_xh_desc(data_w_xh_shape, FORMAT_ND, DT_FLOAT16);
  data_w_xh.update_input_desc_x(data_w_xh_desc);
  data_w_xh.update_output_desc_y(data_w_xh_desc);

  //init_c
  auto data_bias_h = op::Data("bias_h");
  std::vector<int64_t> data_bias_h_vec{512};
  ge::Shape data_bias_h_shape(data_bias_h_vec);
  ge::TensorDesc data_bias_h_desc(data_bias_h_shape, FORMAT_ND, DT_FLOAT16);
  data_bias_h.update_input_desc_x(data_bias_h_desc);
  data_bias_h.update_output_desc_y(data_bias_h_desc);

  //h
  auto data_w_sh = op::Data("w_sh");
  std::vector<int64_t> data_w_sh_vec{512, 18};
  ge::Shape data_w_sh_shape(data_w_sh_vec);
  ge::TensorDesc data_w_sh_desc(data_w_sh_shape, FORMAT_ND, DT_FLOAT16);
  data_w_sh.update_input_desc_x(data_w_sh_desc);
  data_w_sh.update_output_desc_y(data_w_sh_desc);

  //c
  auto data_w_hh = op::Data("w_hh");
  std::vector<int64_t> data_w_hh_vec{512, 512};
  ge::Shape data_w_hh_shape(data_w_hh_vec);
  ge::TensorDesc data_w_hh_desc(data_w_hh_shape, FORMAT_ND, DT_FLOAT16);
  data_w_hh.update_input_desc_x(data_w_hh_desc);
  data_w_hh.update_output_desc_y(data_w_hh_desc);

  //dy
  auto data_wh_o = op::Data("wh_o");
  std::vector<int64_t> data_wh_o_vec{512, 512};
  ge::Shape data_wh_o_shape(data_wh_o_vec);
  ge::TensorDesc data_wh_o_desc(data_wh_o_shape, FORMAT_ND, DT_FLOAT16);
  data_wh_o.update_input_desc_x(data_wh_o_desc);
  data_wh_o.update_output_desc_y(data_wh_o_desc);

  //dh
  auto data_bias_o = op::Data("bias_o");
  std::vector<int64_t> data_bias_o_vec{512};
  ge::Shape data_bias_o_shape(data_bias_o_vec);
  ge::TensorDesc data_bias_o_desc(data_bias_o_shape, FORMAT_ND, DT_FLOAT16);
  data_bias_o.update_input_desc_x(data_bias_o_desc);
  data_bias_o.update_output_desc_y(data_bias_o_desc);

  
  auto rnn_op = op::RNN("RNN");
  rnn_op.set_input_x(data_x)
        .set_input_cont(data_cont)
        .set_input_x_static(data_x_static)
        .set_input_h_0(data_h_0)
        .set_input_w_xh(data_w_xh)
        .set_input_bias_h(data_bias_h)
        .set_input_w_sh(data_w_sh)
        .set_input_w_hh(data_w_hh)
        .set_input_w_ho(data_wh_o)
        .set_input_bias_o(data_bias_o);

  std::vector<Operator> inputs{data_x, data_cont, data_x_static, data_h_0, data_w_xh, data_bias_h,
                               data_w_sh, data_w_hh, data_wh_o, data_bias_o};
  std::vector<Operator> outputs{rnn_op};

  ge::Graph graph("rnn_fusion_test_1");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("RNNFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findBasicRNNCell = false;
  bool findConcat = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "BasicRNNCell") {
            findBasicRNNCell = true;
        }

        if (node->GetType() == "ConcatD") {
            findConcat = true;
        }

    }

  EXPECT_EQ(findBasicRNNCell, true);
  EXPECT_EQ(findConcat, true);
}
