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

class swin_transformer_ln_qkv_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "swin_transformer_ln_qkv_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "swin_transformer_ln_qkv_fusion_test TearDown" << std::endl;
  }
};

TEST_F(swin_transformer_ln_qkv_fusion_test, swin_transformer_ln_qkv_fusion_test_1) {
    // success testcase
    ge::Graph graph("swin_transformer_ln_qkv_fusion_test_1");
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

    TensorDesc graph_input_data_desc(ge::Shape({batch_num, m_num, k_num}), FORMAT_ND, DT_FLOAT);

    graph_input_data_node.update_input_desc_x(graph_input_data_desc);
    graph_input_data_node.update_output_desc_y(graph_input_data_desc);

    // add layer norm
    auto layer_norm_node = op::LayerNorm("layer_norm_0");

    TensorDesc layer_norm_input_desc_0(ge::Shape({batch_num, m_num, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc layer_norm_input_desc_1(ge::Shape({k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc layer_norm_input_desc_2(ge::Shape({k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc layer_norm_output_desc_0(ge::Shape({batch_num, m_num, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc layer_norm_output_desc_1(ge::Shape({batch_num, m_num, 1}), FORMAT_ND, DT_FLOAT);
    TensorDesc layer_norm_output_desc_2(ge::Shape({batch_num, m_num, 1}), FORMAT_ND, DT_FLOAT);

    Tensor layer_norm_input_tensor_1(layer_norm_input_desc_1);
    auto layer_norm_input_data_1 = op::Const("layer_norm_input_data_1").set_attr_value(layer_norm_input_tensor_1);
    Tensor layer_norm_input_tensor_2(layer_norm_input_desc_2);
    auto layer_norm_input_data_2 = op::Const("layer_norm_input_data_2").set_attr_value(layer_norm_input_tensor_2);

    layer_norm_node.update_input_desc_x(layer_norm_input_desc_0);
    layer_norm_node.update_input_desc_gamma(layer_norm_input_desc_1);
    layer_norm_node.update_input_desc_beta(layer_norm_input_desc_2);
    layer_norm_node.update_output_desc_y(layer_norm_output_desc_0);
    layer_norm_node.update_output_desc_mean(layer_norm_output_desc_1);
    layer_norm_node.update_output_desc_variance(layer_norm_output_desc_2);

    layer_norm_node.set_input_x(graph_input_data_node, "y");
    layer_norm_node.set_input_gamma(layer_norm_input_data_1);
    layer_norm_node.set_input_beta(layer_norm_input_data_2);
    layer_norm_node.set_attr_begin_norm_axis(2);
    layer_norm_node.set_attr_begin_params_axis(-1);

    // add reshape_0
    auto reshape_0 = op::Reshape("reshape_0");
    TensorDesc reshape_0_input_desc_0(ge::Shape({batch_num, m_num, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc reshape_0_input_desc_1(ge::Shape({4}), FORMAT_ND, DT_INT32);
    TensorDesc reshape_0_output_desc_0(ge::Shape({batch_num, m_2 * m_3, m_2 * m_3, k_num}), FORMAT_ND, DT_FLOAT);

    Tensor reshape_0_input_tensor_1(reshape_0_input_desc_1);
    auto reshape_0_input_data_1 = op::Const("reshape_0_input_data_1").set_attr_value(reshape_0_input_tensor_1);

    reshape_0.update_input_desc_x(reshape_0_input_desc_0);
    reshape_0.update_input_desc_shape(reshape_0_input_desc_1);
    reshape_0.update_output_desc_y(reshape_0_output_desc_0);

    reshape_0.set_input_x(layer_norm_node, "y");
    reshape_0.set_input_shape(reshape_0_input_data_1);

    // add slice_0
    auto slice_0 = op::StridedSliceD("slice_0");
    TensorDesc slice_0_input_desc_0(ge::Shape({batch_num, m_2 * m_3, m_2 * m_3, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc slice_0_output_desc_0(ge::Shape({batch_num, m_2 * m_3 - roll_num, m_2 * m_3, k_num}), FORMAT_ND, DT_FLOAT);

    slice_0.update_input_desc_x(slice_0_input_desc_0);
    slice_0.update_output_desc_y(slice_0_output_desc_0);

    slice_0.set_input_x(reshape_0, "y");
    slice_0.set_attr_begin({0, roll_num, 0, 0});
    slice_0.set_attr_end({batch_num, m_2 * m_3, m_2 * m_3, k_num});
    slice_0.set_attr_strides({1, 1, 1, 1});

    // add slice_1
    auto slice_1 = op::StridedSliceD("slice_1");
    TensorDesc slice_1_input_desc_0(ge::Shape({batch_num, m_2 * m_3, m_2 * m_3, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc slice_1_output_desc_0(ge::Shape({batch_num, roll_num, m_2 * m_3, k_num}), FORMAT_ND, DT_FLOAT);

    slice_1.update_input_desc_x(slice_1_input_desc_0);
    slice_1.update_output_desc_y(slice_1_output_desc_0);

    slice_1.set_input_x(reshape_0, "y");
    slice_1.set_attr_begin({0, 0, 0, 0});
    slice_1.set_attr_end({batch_num, roll_num, m_2 * m_3, k_num});
    slice_1.set_attr_strides({1, 1, 1, 1});

    // add concat
    auto concat_0 = op::ConcatD("concat_0");
    TensorDesc concat_0_input_desc_0(ge::Shape({batch_num, m_2 * m_3 - roll_num, m_2 * m_3, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc concat_0_input_desc_1(ge::Shape({batch_num, roll_num, m_2 * m_3, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc concat_0_output_desc_0(ge::Shape({batch_num, m_2 * m_3, m_2 * m_3, k_num}), FORMAT_ND, DT_FLOAT);

    concat_0.create_dynamic_input_x(2);
    concat_0.update_dynamic_input_desc_x(0, concat_0_input_desc_0);
    concat_0.update_dynamic_input_desc_x(0, concat_0_input_desc_1);
    concat_0.update_output_desc_y(concat_0_output_desc_0);

    concat_0.set_dynamic_input_x(0, slice_0, "y");
    concat_0.set_dynamic_input_x(1, slice_1, "y");
    concat_0.set_attr_concat_dim(1);
    concat_0.set_attr_N(2);

    // add slice_2
    auto slice_2 = op::StridedSliceD("slice_2");
    TensorDesc slice_2_input_desc_0(ge::Shape({batch_num, m_2 * m_3, m_2 * m_3, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc slice_2_output_desc_0(ge::Shape({batch_num, m_2 * m_3, m_2 * m_3 - roll_num, k_num}), FORMAT_ND, DT_FLOAT);

    slice_2.update_input_desc_x(slice_2_input_desc_0);
    slice_2.update_output_desc_y(slice_2_output_desc_0);

    slice_2.set_input_x(concat_0, "y");
    slice_2.set_attr_begin({0, 0, roll_num, 0});
    slice_2.set_attr_end({batch_num, m_2 * m_3, m_2 * m_3, k_num});
    slice_2.set_attr_strides({1, 1, 1, 1});

    // add slice_3
    auto slice_3 = op::StridedSliceD("slice_3");
    TensorDesc slice_3_input_desc_0(ge::Shape({batch_num, m_2 * m_3, m_2 * m_3, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc slice_3_output_desc_0(ge::Shape({batch_num, m_2 * m_3, roll_num, k_num}), FORMAT_ND, DT_FLOAT);

    slice_3.update_input_desc_x(slice_3_input_desc_0);
    slice_3.update_output_desc_y(slice_3_output_desc_0);

    slice_3.set_input_x(concat_0, "y");
    slice_3.set_attr_begin({0, 0, 0, 0});
    slice_3.set_attr_end({batch_num, m_2 * m_3, roll_num, k_num});
    slice_3.set_attr_strides({1, 1, 1, 1});

    // add concat_1
    auto concat_1 = op::ConcatD("concat_1");
    TensorDesc concat_1_input_desc_0(ge::Shape({batch_num, m_2 * m_3, m_2 * m_3 - roll_num, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc concat_1_input_desc_1(ge::Shape({batch_num, m_2 * m_3, roll_num, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc concat_1_desc_out_0(ge::Shape({batch_num, m_2 * m_3, m_2 * m_3, k_num}), FORMAT_ND, DT_FLOAT);

    concat_1.create_dynamic_input_x(2);
    concat_1.update_dynamic_input_desc_x(0, concat_1_input_desc_0);
    concat_1.update_dynamic_input_desc_x(0, concat_1_input_desc_1);
    concat_1.update_output_desc_y(concat_1_desc_out_0);

    concat_1.set_dynamic_input_x(0, slice_2, "y");
    concat_1.set_dynamic_input_x(1, slice_3, "y");
    concat_1.set_attr_concat_dim(2);
    concat_1.set_attr_N(2);

    // reshape_1
    auto reshape_1 = op::Reshape("reshape_1");

    TensorDesc reshape_1_input_desc_0(ge::Shape({batch_num, m_2 * m_3, m_2 * m_3, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc reshape_1_input_desc_1(ge::Shape({6}), FORMAT_ND, DT_INT32);
    TensorDesc reshape_1_output_desc_0(ge::Shape({batch_num, m_2, m_3, m_2, m_3, k_num}), FORMAT_ND, DT_FLOAT);

    Tensor reshape_1_input_tensor_1(reshape_1_input_desc_1);
    auto reshape_1_input_data_1 = op::Const("reshape_1_input_data_1").set_attr_value(reshape_1_input_tensor_1);

    reshape_1.update_input_desc_x(reshape_1_input_desc_0);
    reshape_1.update_input_desc_shape(reshape_1_input_desc_1);
    reshape_1.update_output_desc_y(reshape_1_output_desc_0);
    reshape_1.set_input_x(concat_1, "y");
    reshape_1.set_input_shape(reshape_1_input_data_1);

    // add transpose_0
    auto transpose_0 = op::TransposeD("transpose_0");
    TensorDesc transpose_0_input_desc_0(ge::Shape({batch_num, m_2, m_3, m_2, m_3, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc transpose_0_output_desc_0(ge::Shape({batch_num, m_2, m_2, m_3, m_3, k_num}), FORMAT_ND, DT_FLOAT);

    transpose_0.update_input_desc_x(transpose_0_input_desc_0);
    transpose_0.update_output_desc_y(transpose_0_output_desc_0);

    transpose_0.set_input_x(reshape_1, "y");
    transpose_0.set_attr_perm({0, 1, 3, 2, 4, 5});

    // reshape_2
    auto reshape_2 = op::Reshape("reshape_2");
    TensorDesc reshape_2_input_desc_0(ge::Shape({batch_num, m_2, m_2, m_3, m_3, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc reshape_2_input_desc_1(ge::Shape({4}), FORMAT_ND, DT_INT32);
    TensorDesc reshape_2_output_desc_0(ge::Shape({batch_num * m_2 * m_2, m_3, m_3, k_num}), FORMAT_ND, DT_FLOAT);

    Tensor reshape_2_input_tensor_1(reshape_2_input_desc_1);
    auto reshape_2_input_data_1 = op::Const("reshape_2_input_data_1").set_attr_value(reshape_2_input_tensor_1);

    reshape_2.update_input_desc_x(reshape_2_input_desc_0);
    reshape_2.update_input_desc_shape(reshape_2_input_desc_1);
    reshape_2.update_output_desc_y(reshape_2_output_desc_0);

    reshape_2.set_input_x(transpose_0, "y");
    reshape_2.set_input_shape(reshape_2_input_data_1);

    // reshape_3
    auto reshape_3 = op::Reshape("reshape_3");
    TensorDesc reshape_3_input_desc_0(ge::Shape({batch_num * m_2 * m_2, m_3, m_3, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc reshape_3_input_desc_1(ge::Shape({3}), FORMAT_ND, DT_INT32);
    TensorDesc reshape_3_output_desc_0(ge::Shape({batch_num * m_2 * m_2, m_3 * m_3, k_num}), FORMAT_ND, DT_FLOAT);

    Tensor reshape_3_input_tensor_1(reshape_3_input_desc_1);
    auto reshape_3_input_data_1 = op::Const("reshape_3_input_data_1").set_attr_value(reshape_3_input_tensor_1);

    reshape_3.update_input_desc_x(reshape_3_input_desc_0);
    reshape_3.update_input_desc_shape(reshape_3_input_desc_1);
    reshape_3.update_output_desc_y(reshape_3_output_desc_0);

    reshape_3.set_input_x(reshape_2, "y");
    reshape_3.set_input_shape(reshape_3_input_data_1);

    // batch_matmul
    auto batch_matmul = op::BatchMatMulV2("batch_matmul");
    TensorDesc batch_matmul_input_desc_0(ge::Shape({batch_num * m_2 * m_2, m_3 * m_3, k_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc batch_matmul_input_desc_1(ge::Shape({k_num, n_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc batch_matmul_input_desc_2(ge::Shape({n_num, }), FORMAT_ND, DT_FLOAT);
    TensorDesc batch_matmul_output_desc_0(ge::Shape({batch_num * m_2 * m_2, m_3 * m_3, n_num}), FORMAT_ND, DT_FLOAT);

    Tensor batch_matmul_input_tensor_1(batch_matmul_input_desc_1);
    auto batch_matmul_input_data_1 = op::Const("batch_matmul_input_data_1").set_attr_value(batch_matmul_input_tensor_1);

    Tensor batch_matmul_input_tensor_2(batch_matmul_input_desc_2);
    auto batch_matmul_input_data_2 = op::Const("batch_matmul_input_data_2").set_attr_value(batch_matmul_input_tensor_2);

    batch_matmul.update_input_desc_x1(batch_matmul_input_desc_0);
    batch_matmul.update_input_desc_x2(batch_matmul_input_desc_1);
    batch_matmul.update_input_desc_bias(batch_matmul_input_desc_2);
    batch_matmul.update_output_desc_y(batch_matmul_output_desc_0);

    batch_matmul.set_input_x1(reshape_3, "y");
    batch_matmul.set_input_x2(batch_matmul_input_data_1);
    batch_matmul.set_input_bias(batch_matmul_input_data_2);

    // confuse_0
    auto confuse_0 = op::ConfusionTransposeD("confuse_0");
    TensorDesc confuse_0_input_desc_0(ge::Shape({batch_num * m_2 * m_2, m_3 * m_3, n_num}), FORMAT_ND, DT_FLOAT);
    TensorDesc confuse_0_output_desc_0(ge::Shape({3, batch_num * m_2 * m_2, n_num / 3 / head_dim, m_3 * m_3, head_dim}), FORMAT_ND, DT_FLOAT);

    confuse_0.update_input_desc_x(confuse_0_input_desc_0);
    confuse_0.update_output_desc_y(confuse_0_output_desc_0);

    confuse_0.set_input_x(batch_matmul, "y");
    confuse_0.set_attr_perm({2, 0, 3, 1, 4});
    confuse_0.set_attr_shape({batch_num * m_2 * m_2, m_3 * m_3, 3, n_num / 3 / head_dim, head_dim});

    // split_0
    auto split_0 = op::SplitVD("split_0");
    TensorDesc split_0_input_desc_0(ge::Shape({3, batch_num * m_2 * m_2, n_num / 3 / head_dim, m_3 * m_3, head_dim}), FORMAT_ND, DT_FLOAT);
    TensorDesc split_0_output_desc_0(ge::Shape({1, batch_num * m_2 * m_2, n_num / 3 / head_dim, m_3 * m_3, head_dim}), FORMAT_ND, DT_FLOAT);

    split_0.update_input_desc_x(split_0_input_desc_0);
    split_0.create_dynamic_output_y(3);
    split_0.update_dynamic_output_desc_y(0, split_0_output_desc_0);
    split_0.update_dynamic_output_desc_y(1, split_0_output_desc_0);
    split_0.update_dynamic_output_desc_y(2, split_0_output_desc_0);

    split_0.set_input_x(confuse_0, "y");
    split_0.set_attr_size_splits({1, 1, 1});
    split_0.set_attr_split_dim(0);
    split_0.set_attr_num_split(3);

    // squeeze_0
    auto squeeze_0 = op::Squeeze("squeeze_0");
    TensorDesc squeeze_0_input_desc_0(ge::Shape({1, batch_num * m_2 * m_2, n_num / 3 / head_dim, m_3 * m_3, head_dim}), FORMAT_ND, DT_FLOAT);
    TensorDesc squeeze_0_output_desc_0(ge::Shape({batch_num * m_2 * m_2, n_num / 3 / head_dim, m_3 * m_3, head_dim}), FORMAT_ND, DT_FLOAT);

    squeeze_0.update_input_desc_x(squeeze_0_input_desc_0);
    squeeze_0.update_output_desc_y(squeeze_0_output_desc_0);

    squeeze_0.set_input_x(split_0, 0);
    ge::Operator::OpListInt squeeze_0_axis = {0};
    squeeze_0.set_attr_axis(squeeze_0_axis);

    // squeeze_0
    auto squeeze_1 = op::Squeeze("squeeze_1");
    TensorDesc squeeze_1_input_desc_0(ge::Shape({1, batch_num * m_2 * m_2, n_num / 3 / head_dim, m_3 * m_3, head_dim}), FORMAT_ND, DT_FLOAT);
    TensorDesc squeeze_1_output_desc_0(ge::Shape({batch_num * m_2 * m_2, n_num / 3 / head_dim, m_3 * m_3, head_dim}), FORMAT_ND, DT_FLOAT);

    squeeze_1.update_input_desc_x(squeeze_1_input_desc_0);
    squeeze_1.update_output_desc_y(squeeze_1_output_desc_0);

    squeeze_1.set_input_x(split_0, 1);
    ge::Operator::OpListInt squeeze_1_axis = {0};
    squeeze_1.set_attr_axis(squeeze_1_axis);

    // squeeze_0
    auto squeeze_2 = op::Squeeze("squeeze_2");
    TensorDesc squeeze_2_input_desc_0(ge::Shape({1, batch_num * m_2 * m_2, n_num / 3 / head_dim, m_3 * m_3, head_dim}), FORMAT_ND, DT_FLOAT);
    TensorDesc squeeze_2_output_desc_0(ge::Shape({batch_num * m_2 * m_2, n_num / 3 / head_dim, m_3 * m_3, head_dim}), FORMAT_ND, DT_FLOAT);

    squeeze_2.update_input_desc_x(squeeze_2_input_desc_0);
    squeeze_2.update_output_desc_y(squeeze_2_output_desc_0);

    squeeze_2.set_input_x(split_0, 2);
    ge::Operator::OpListInt squeeze_2_axis = {0};
    squeeze_2.set_attr_axis(squeeze_2_axis);

    // start fusion
    std::vector<Operator> inputs{graph_input_data_node};
    std::vector<Operator> outputs{squeeze_0, squeeze_1, squeeze_2};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("SwinTransformerLnQKVFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool find_op = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SwinTransformerLnQKV") {
            find_op = true;
            break;
        }
    }
    EXPECT_EQ(find_op, true);
}
