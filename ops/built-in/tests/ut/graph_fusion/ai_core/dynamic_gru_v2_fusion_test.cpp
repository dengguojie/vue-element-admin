#include "gtest/gtest.h"
// #include <mockcpp/mockcpp.hpp>
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "rnn.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;

uint32_t GetPlatformInfoWithOutSocVersionStub(fe::PlatformInfoManager *This,
                                              fe::PlatformInfo &platform_info,
                                              fe::OptionalInfo &opti_compilation_info) {
  platform_info.ai_core_spec.l1_size = 1048576;
  platform_info.soc_info.ai_core_cnt = 32;
  return ge::GRAPH_SUCCESS;
}

class dynamic_gru_v2_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_gru_v2_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dynamic_gru_v2_fusion_test TearDown" << std::endl;
    }
};

// NOTE: we cannot use mockcpp here, using gmock instead.
// TEST_F(dynamic_gru_v2_fusion_test, dynamic_gru_v2_fusion_test_1) {
//   MOCKER_CPP(&fe::PlatformInfoManager::GetPlatformInfoWithOutSocVersion).stubs().will(invoke(GetPlatformInfoWithOutSocVersionStub));
// 
//   // x
//   auto data_x = op::Data("x");
//   std::vector<int64_t> data_x_vec{50,32,1024};
//   ge::Shape data_x_shape(data_x_vec);
//   ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT16);
//   data_x.update_input_desc_x(data_x_desc);
//   data_x.update_output_desc_y(data_x_desc);
// 
//   // weight_input
//   auto data_weight_input = op::Data("weight_input");
//   std::vector<int64_t> data_weight_input_vec{1024,1536};
//   ge::Shape data_weight_input_shape(data_weight_input_vec);
//   ge::TensorDesc data_weight_input_desc(data_weight_input_shape, FORMAT_ND, DT_FLOAT16);
//   data_weight_input.update_input_desc_x(data_weight_input_desc);
//   data_weight_input.update_output_desc_y(data_weight_input_desc);
//   
//   // weight_hidden
//   auto data_weight_hidden = op::Data("weight_hidden");
//   std::vector<int64_t> data_weight_hidden_vec{512,1536};
//   ge::Shape data_weight_hidden_shape(data_weight_hidden_vec);
//   ge::TensorDesc data_weight_hidden_desc(data_weight_hidden_shape, FORMAT_ND, DT_FLOAT16);
//   data_weight_hidden.update_input_desc_x(data_weight_hidden_desc);
//   data_weight_hidden.update_output_desc_y(data_weight_hidden_desc);
//   
//   // bias_input
//   auto data_bias_input = op::Data("bias_input");
//   std::vector<int64_t> data_bias_input_vec{1536};
//   ge::Shape data_bias_input_shape(data_bias_input_vec);
//   ge::TensorDesc data_bias_input_desc(data_bias_input_shape, FORMAT_ND, DT_FLOAT);
//   data_bias_input.update_input_desc_x(data_bias_input_desc);
//   data_bias_input.update_output_desc_y(data_bias_input_desc);
// 
//   // bias_hidden
//   auto data_bias_hidden = op::Data("bias_hidden");
//   std::vector<int64_t> data_bias_hidden_vec{1536};
//   ge::Shape data_bias_hidden_shape(data_bias_hidden_vec);
//   ge::TensorDesc data_bias_hidden_desc(data_bias_hidden_shape, FORMAT_ND, DT_FLOAT);
//   data_bias_hidden.update_input_desc_x(data_bias_hidden_desc);
//   data_bias_hidden.update_output_desc_y(data_bias_hidden_desc);
// 
//   // init_h
//   auto data_init_h = op::Data("init_h");
//   std::vector<int64_t> data_init_h_vec{32,512};
//   ge::Shape data_init_h_shape(data_init_h_vec);
//   ge::TensorDesc data_init_h_desc(data_init_h_shape, FORMAT_ND, DT_FLOAT);
//   data_init_h.update_input_desc_x(data_init_h_desc);
//   data_init_h.update_output_desc_y(data_init_h_desc);
// 
//   auto gru_v2_op = op::DynamicGRUV2("DynamicGRUV2");
//   gru_v2_op.set_input_x(data_x)
//            .set_input_weight_input(data_weight_input)
//            .set_input_weight_hidden(data_weight_hidden)
//            .set_input_bias_input(data_bias_input)
//            .set_input_bias_hidden(data_bias_hidden)
//            .set_input_init_h(data_init_h);
//            
//   std::vector<Operator> inputs{data_x, data_weight_input, data_weight_hidden, data_bias_input, data_bias_hidden, data_init_h};
//   std::vector<Operator> outputs{gru_v2_op};
// 
//   ge::Graph graph("dynamic_gru_v2_fusion_test");
//   graph.SetInputs(inputs).SetOutputs(outputs);
// 
//   ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
//   fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
//   fe::FusionPassTestUtils::RunGraphFusionPass("DynamicGRUV2FusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
// 
//   bool find_gru_hidden = false;
//   bool find_matmul = false;
//   bool find_gruv2 = false;
//   for (auto node: compute_graph_ptr->GetAllNodes()) {
//     if (node->GetType() == "BatchMatMulV2") {
//       find_matmul = true;
//     }
// 
//     if (node->GetType() == "DynamicGRUV2Hidden") {
//       find_gru_hidden = true;
//     }
// 
//     if (node->GetType() == "DynamicGRUV2") {
//       find_gruv2 = true;
//     }
//   }
// 
//   EXPECT_EQ(find_gru_hidden, true);
//   EXPECT_EQ(find_matmul, true);
//   EXPECT_EQ(find_gruv2, false);
// }
