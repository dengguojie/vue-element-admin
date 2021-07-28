#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nonlinear_fuc_ops.h"
#include "array_ops.h"
#include "nn_pooling_ops.h"
#include "fusion_pass_test_utils.h"
#include "nn_calculation_ops.h"
#include "all_ops.h"

#define private public
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;

class resnet50_dbn_dw_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "resnet50_dbn_dw_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "resnet50_dbn_dw_fusion_test TearDown" << std::endl;
    }
};

TEST_F(resnet50_dbn_dw_fusion_test, resnet50_dbn_dw_fusion_test_1) {
    ge::Graph graph("resnet50_dbn_dw_fusion_test_1");

    auto dedx_shape = vector<int64_t>({32, 64, 56, 56});
    ge::TensorDesc desc_dedx(ge::Shape(dedx_shape), FORMAT_NCHW, DT_FLOAT16);
    auto dedy_shape = vector<int64_t>({32, 64, 56, 56});
    ge::TensorDesc desc_dedy(ge::Shape(dedy_shape), FORMAT_NCHW, DT_FLOAT16);
    auto channel_shape = ge::Shape({1,64,1,1});
    ge::TensorDesc desc_channel(ge::Shape(channel_shape), FORMAT_NCHW, DT_FLOAT);

    auto data_dx = op::Data("data_dx").set_attr_index(0);
    data_dx.update_input_desc_x(desc_dedx);
    data_dx.update_output_desc_y(desc_dedx);

    auto data_dy = op::Data("data_dy").set_attr_index(1);
    data_dy.update_input_desc_x(desc_dedy);
    data_dy.update_output_desc_y(desc_dedy);
    auto data_dbnx = op::Data("data_dbnx").set_attr_index(2);
    data_dbnx.update_input_desc_x(desc_dedy);
    data_dbnx.update_output_desc_y(desc_dedy);

    auto data_diff_scale = op::Data("data_diff_scale").set_attr_index(3);
    data_diff_scale.update_input_desc_x(desc_channel);
    data_diff_scale.update_output_desc_y(desc_channel);
    auto data_diff_offset = op::Data("data_diff_offset").set_attr_index(4);
    data_diff_offset.update_input_desc_x(desc_channel);
    data_diff_offset.update_output_desc_y(desc_channel);
    auto data_scale = op::Data("data_scale").set_attr_index(5);
    data_scale.update_input_desc_x(desc_channel);
    data_scale.update_output_desc_y(desc_channel);
    auto data_batch_mean = op::Data("data_batch_mean").set_attr_index(6);
    data_batch_mean.update_input_desc_x(desc_channel);
    data_batch_mean.update_output_desc_y(desc_channel);
    auto data_batch_variance = op::Data("data_batch_variance").set_attr_index(7);
    data_batch_variance.update_input_desc_x(desc_channel);
    data_batch_variance.update_output_desc_y(desc_channel);

    auto dbnOp = op::BNTrainingReduceGrad("dbn")
        .set_input_grads(data_dy)
        .set_input_x(data_dbnx)
        .set_input_diff_scale(data_diff_scale)
        .set_input_diff_offset(data_diff_offset)
        .set_input_scale(data_scale)
        .set_input_batch_mean(data_batch_mean)
        .set_input_batch_variance(data_batch_variance)
        .set_attr_epsilon({0.0001});
    dbnOp.update_output_desc_y(desc_dedy);

    auto conv2dbackpropfilterd = op::Conv2DBackpropFilterD("conv2dbackpropfilterd")
        .set_input_x(data_dx)
        .set_input_out_backprop(dbnOp)
        .set_attr_filter_size({64,64,1,1})
        .set_attr_strides({1,1,1,1})
        .set_attr_pads({0,0,0,0})
        .set_attr_dilations({1,1,1,1})
        .set_attr_groups({1})
        .set_attr_data_format("NCHW");
    auto dw_shape = ge::Shape({64,64,1,1});
    ge::TensorDesc desc_dw(ge::Shape(dw_shape), FORMAT_NCHW, DT_FLOAT);
    conv2dbackpropfilterd.update_output_desc_y(desc_dw);

    std::vector<Operator> inputs{data_dx, data_dy, data_dbnx, data_diff_scale, data_diff_offset, data_scale, data_batch_mean, data_batch_variance};
    std::vector<Operator> outputs{conv2dbackpropfilterd};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "Ascend910A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("Resnet50DbnDwFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    fe::PlatformInfoManager::Instance().platform_info_map_.clear();

    bool findFusedNode = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "FusedDbnDw") {
            findFusedNode = true;
        }
    }
    EXPECT_EQ(findFusedNode, true);
}

TEST_F(resnet50_dbn_dw_fusion_test, resnet50_dbn_dw_fusion_test_2) {
    ge::Graph graph("resnet50_dbn_dw_fusion_test_2");

    auto dedx_shape = vector<int64_t>({256, 256, 14, 14});
    ge::TensorDesc desc_dedx(ge::Shape(dedx_shape), FORMAT_NCHW, DT_FLOAT16);
    auto dedy_shape = vector<int64_t>({256, 1024, 14, 14});
    ge::TensorDesc desc_dedy(ge::Shape(dedy_shape), FORMAT_NCHW, DT_FLOAT16);
    auto channel_shape = ge::Shape({1, 1024, 1, 1});
    ge::TensorDesc desc_channel(ge::Shape(channel_shape), FORMAT_NCHW, DT_FLOAT);

    auto data_dx = op::Data("data_dx").set_attr_index(0);
    data_dx.update_input_desc_x(desc_dedx);
    data_dx.update_output_desc_y(desc_dedx);

    auto data_dy = op::Data("data_dy").set_attr_index(1);
    data_dy.update_input_desc_x(desc_dedy);
    data_dy.update_output_desc_y(desc_dedy);
    auto data_dbnx = op::Data("data_dbnx").set_attr_index(2);
    data_dbnx.update_input_desc_x(desc_dedy);
    data_dbnx.update_output_desc_y(desc_dedy);

    auto data_diff_scale = op::Data("data_diff_scale").set_attr_index(3);
    data_diff_scale.update_input_desc_x(desc_channel);
    data_diff_scale.update_output_desc_y(desc_channel);
    auto data_diff_offset = op::Data("data_diff_offset").set_attr_index(4);
    data_diff_offset.update_input_desc_x(desc_channel);
    data_diff_offset.update_output_desc_y(desc_channel);
    auto data_scale = op::Data("data_scale").set_attr_index(5);
    data_scale.update_input_desc_x(desc_channel);
    data_scale.update_output_desc_y(desc_channel);
    auto data_batch_mean = op::Data("data_batch_mean").set_attr_index(6);
    data_batch_mean.update_input_desc_x(desc_channel);
    data_batch_mean.update_output_desc_y(desc_channel);
    auto data_batch_variance = op::Data("data_batch_variance").set_attr_index(7);
    data_batch_variance.update_input_desc_x(desc_channel);
    data_batch_variance.update_output_desc_y(desc_channel);

    auto dbnOp = op::BNTrainingReduceGrad("dbn")
        .set_input_grads(data_dy)
        .set_input_x(data_dbnx)
        .set_input_diff_scale(data_diff_scale)
        .set_input_diff_offset(data_diff_offset)
        .set_input_scale(data_scale)
        .set_input_batch_mean(data_batch_mean)
        .set_input_batch_variance(data_batch_variance)
        .set_attr_epsilon({0.0001});
    dbnOp.update_output_desc_y(desc_dedy);

    auto conv2dbackpropfilterd = op::Conv2DBackpropFilterD("conv2dbackpropfilterd")
        .set_input_x(data_dx)
        .set_input_out_backprop(dbnOp)
        .set_attr_filter_size({1024, 256, 1, 1})
        .set_attr_strides({1,1,1,1})
        .set_attr_pads({0,0,0,0})
        .set_attr_dilations({1,1,1,1})
        .set_attr_groups({1})
        .set_attr_data_format("NCHW");
    auto dw_shape = ge::Shape({1024,256,1,1});
    ge::TensorDesc desc_dw(ge::Shape(dw_shape), FORMAT_NCHW, DT_FLOAT);
    conv2dbackpropfilterd.update_output_desc_y(desc_dw);

    std::vector<Operator> inputs{data_dx, data_dy, data_dbnx, data_diff_scale, data_diff_offset, data_scale, data_batch_mean, data_batch_variance};
    std::vector<Operator> outputs{conv2dbackpropfilterd};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "Ascend910A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("Resnet50DbnDwFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    fe::PlatformInfoManager::Instance().platform_info_map_.clear();

    bool findFusedNode = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "FusedDbnDw") {
            findFusedNode = true;
        }
    }
    EXPECT_EQ(findFusedNode, true);
}