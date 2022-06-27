#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "matrix_calculation_ops.h"
#include "nn_norm_ops.h"
#include "selection_ops.h"
#include "split_combination_ops.h"
#include "transformation_ops.h"
#include "nonlinear_fuc_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/operator_reg.h"
#include "fc_transdata_merge_fusion_pass_test.h"
#define private public
#include "common/util/platform_info.h"


using namespace ge;
using namespace op;

class swin_attention_ffn_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "swin_attention_ffn_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "swin_attention_ffn_fusion_test TearDown" << std::endl;
  }
};

TEST_F(swin_attention_ffn_fusion_test, swin_attention_ffn_fusion_test_1) {
    // success testcase
    ge::Graph graph("swin_attention_ffn_fusion_test_1");
    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 8;
    platform_info.str_info.short_soc_version = "Ascend310P";
    opti_compilation_info.soc_version = "Ascend310P3";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend310P3"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    
    int64_t batch_num = 8;
    int64_t m_2 = 8;
    int64_t m_3 = 12;
    int64_t m_num = m_2 * m_3 * m_2 * m_3;
    int64_t k_num = 128;
    int64_t n_num = 384;
    int64_t seq_length = m_2 * m_2;
    int64_t head_dim = 32;
    int64_t roll_num = 6;
    // add input data
    auto graph_input_data_node = op::Data("graph_input_data_node");

    TensorDesc graph_input_data_desc(ge::Shape({512, 144, 128}), FORMAT_ND, DT_FLOAT);

    graph_input_data_node.update_input_desc_x(graph_input_data_desc);
    graph_input_data_node.update_output_desc_y(graph_input_data_desc);

    // batch_matmul
    auto batch_matmul = op::BatchMatMulV2("batch_matmul");
    TensorDesc batch_matmul_input_desc_0(ge::Shape({512, 144, 128}), FORMAT_ND, DT_FLOAT);
    TensorDesc batch_matmul_input_desc_1(ge::Shape({128, 128}), FORMAT_ND, DT_FLOAT);
    TensorDesc batch_matmul_output_desc_0(ge::Shape({512, 144, 128}), FORMAT_ND, DT_FLOAT);

    Tensor batch_matmul_input_tensor_1(batch_matmul_input_desc_1);
    auto batch_matmul_input_data_1 = op::Const("batch_matmul_input_data_1").set_attr_value(batch_matmul_input_tensor_1);

    batch_matmul.update_input_desc_x1(batch_matmul_input_desc_0);
    batch_matmul.update_input_desc_x2(batch_matmul_input_desc_1);
    batch_matmul.update_output_desc_y(batch_matmul_output_desc_0);

    batch_matmul.set_input_x1(graph_input_data_node, "y");
    batch_matmul.set_input_x2(batch_matmul_input_data_1);

    // add
    auto add_0 = op::Add("add_0");
    TensorDesc add_0_input_desc_0(ge::Shape({128}), FORMAT_ND, DT_FLOAT);
    TensorDesc add_0_input_desc_1(ge::Shape({512, 144, 128}), FORMAT_ND, DT_FLOAT);
    TensorDesc add_0_output_desc_0(ge::Shape({512, 144, 128}), FORMAT_ND, DT_FLOAT);

    Tensor add_0_input_tensor_0(add_0_input_desc_0);
    auto add_0_input_data_0 = op::Const("add_0_input_data_1").set_attr_value(add_0_input_tensor_0);

    add_0.update_input_desc_x1(add_0_input_desc_0);
    add_0.update_input_desc_x2(add_0_input_desc_1);
    add_0.update_output_desc_y(add_0_output_desc_0);

    add_0.set_input_x1(add_0_input_data_0);
    add_0.set_input_x2(batch_matmul, "y");
    // add reshape_0
    auto reshape_0 = op::Reshape("reshape_0");
    TensorDesc reshape_0_input_desc_0(ge::Shape({512, 144, 128}), FORMAT_ND, DT_FLOAT);
    TensorDesc reshape_0_input_desc_1(ge::Shape({4}), FORMAT_ND, DT_INT32);
    TensorDesc reshape_0_output_desc_0(ge::Shape({512, 12, 12, 128}), FORMAT_ND, DT_FLOAT);

    Tensor reshape_0_input_tensor_1(reshape_0_input_desc_1);
    auto reshape_0_input_data_1 = op::Const("reshape_0_input_data_1").set_attr_value(reshape_0_input_tensor_1);

    reshape_0.update_input_desc_x(reshape_0_input_desc_0);
    reshape_0.update_input_desc_shape(reshape_0_input_desc_1);
    reshape_0.update_output_desc_y(reshape_0_output_desc_0);

    reshape_0.set_input_x(add_0, "y");
    reshape_0.set_input_shape(reshape_0_input_data_1);

    // reshape_1
    auto reshape_1 = op::Reshape("reshape_1");

    TensorDesc reshape_1_input_desc_0(ge::Shape({512, 12, 12, 128}), FORMAT_ND, DT_FLOAT);
    TensorDesc reshape_1_input_desc_1(ge::Shape({6}), FORMAT_ND, DT_INT32);
    TensorDesc reshape_1_output_desc_0(ge::Shape({1, 8, 8, 12, 12, 1024}), FORMAT_ND, DT_FLOAT);

    Tensor reshape_1_input_tensor_1(reshape_1_input_desc_1);
    auto reshape_1_input_data_1 = op::Const("reshape_1_input_data_1").set_attr_value(reshape_1_input_tensor_1);

    reshape_1.update_input_desc_x(reshape_1_input_desc_0);
    reshape_1.update_input_desc_shape(reshape_1_input_desc_1);
    reshape_1.update_output_desc_y(reshape_1_output_desc_0);
    reshape_1.set_input_x(reshape_0, "y");
    reshape_1.set_input_shape(reshape_1_input_data_1);

    // add transpose_0
    auto transpose_0 = op::TransposeD("transpose_0");
    TensorDesc transpose_0_input_desc_0(ge::Shape({1, 8, 8, 12, 12, 1024}), FORMAT_ND, DT_FLOAT);
    TensorDesc transpose_0_output_desc_0(ge::Shape({1, 8, 12, 8, 12, 1024}), FORMAT_ND, DT_FLOAT);

    transpose_0.update_input_desc_x(transpose_0_input_desc_0);
    transpose_0.update_output_desc_y(transpose_0_output_desc_0);

    transpose_0.set_input_x(reshape_1, "y");
    transpose_0.set_attr_perm({0, 1, 3, 2, 4, 5});

    // reshape_2
    auto reshape_2 = op::Reshape("reshape_2");
    TensorDesc reshape_2_input_desc_0(ge::Shape({1, 8, 12, 8, 12, 1024}), FORMAT_ND, DT_FLOAT);
    TensorDesc reshape_2_input_desc_1(ge::Shape({4}), FORMAT_ND, DT_INT32);
    TensorDesc reshape_2_output_desc_0(ge::Shape({1, 96, 96, 1024}), FORMAT_ND, DT_FLOAT);

    Tensor reshape_2_input_tensor_1(reshape_2_input_desc_1);
    auto reshape_2_input_data_1 = op::Const("reshape_2_input_data_1").set_attr_value(reshape_2_input_tensor_1);

    reshape_2.update_input_desc_x(reshape_2_input_desc_0);
    reshape_2.update_input_desc_shape(reshape_2_input_desc_1);
    reshape_2.update_output_desc_y(reshape_2_output_desc_0);

    reshape_2.set_input_x(transpose_0, "y");
    reshape_2.set_input_shape(reshape_2_input_data_1);

    // reshape_3
    auto reshape_3 = op::Reshape("reshape_3");
    TensorDesc reshape_3_input_desc_0(ge::Shape({1, 96, 96, 1024}), FORMAT_ND, DT_FLOAT);
    TensorDesc reshape_3_input_desc_1(ge::Shape({3}), FORMAT_ND, DT_INT32);
    TensorDesc reshape_3_output_desc_0(ge::Shape({8, 9216, 128}), FORMAT_ND, DT_FLOAT);

    Tensor reshape_3_input_tensor_1(reshape_3_input_desc_1);
    auto reshape_3_input_data_1 = op::Const("reshape_3_input_data_1").set_attr_value(reshape_3_input_tensor_1);

    reshape_3.update_input_desc_x(reshape_3_input_desc_0);
    reshape_3.update_input_desc_shape(reshape_3_input_desc_1);
    reshape_3.update_output_desc_y(reshape_3_output_desc_0);

    reshape_3.set_input_x(reshape_2, "y");
    reshape_3.set_input_shape(reshape_3_input_data_1);

    // start fusion
    std::vector<Operator> inputs{graph_input_data_node};
    std::vector<Operator> outputs{reshape_3};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("SwinAttentionFFNFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool find_op = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SwinAttentionFFN") {
            find_op = true;
            break;
        }
    }
    EXPECT_EQ(find_op, true);
}
