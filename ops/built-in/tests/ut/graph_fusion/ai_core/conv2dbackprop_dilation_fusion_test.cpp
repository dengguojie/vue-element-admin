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

class conv2dbackprop_dilation_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "conv2dbackprop_dilation_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "conv2dbackprop_dilation_fusion_test TearDown" << std::endl;
    }
};

TEST_F(conv2dbackprop_dilation_fusion_test, conv2dbackprop_dilation_fusion_test_1) {
    ge::Graph graph("conv2dbackprop_dilation_fusion_test_1");

    auto dedw_shape = vector<int64_t>({512, 512, 1, 1});
    ge::TensorDesc desc_dedw(ge::Shape(dedw_shape), FORMAT_NCHW, DT_FLOAT16);
    auto dedy_shape = vector<int64_t>({2, 512, 14, 14});
    ge::TensorDesc desc_dedy(ge::Shape(dedy_shape), FORMAT_NCHW, DT_FLOAT16);

    auto data_dw = op::Data("data_dw").set_attr_index(0);
    data_dw.update_input_desc_x(desc_dedw);
    data_dw.update_output_desc_y(desc_dedw);

    auto data_dy = op::Data("data_dy").set_attr_index(1);
    data_dy.update_input_desc_x(desc_dedy);
    data_dy.update_output_desc_y(desc_dedy);

    auto conv2dbackpropinputd = op::Conv2DBackpropInputD("conv2dbackpropinputd")
        .set_input_filter(data_dw)
        .set_input_out_backprop(data_dy)
        .set_attr_input_size({2, 512, 28, 28})
        .set_attr_strides({1, 1, 2, 2})
        .set_attr_pads({0, 0, 0, 0})
        .set_attr_dilations({1, 1, 1, 1})
        .set_attr_groups({1})
        .set_attr_data_format("NCHW");
    auto out_shape = ge::Shape({2, 512, 28, 28});
    ge::TensorDesc desc_out(ge::Shape(out_shape), FORMAT_NCHW, DT_FLOAT16);
    conv2dbackpropinputd.update_output_desc_y(desc_out);

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(conv2dbackpropinputd);

    std::vector<Operator> inputs{data_dw, data_dy};
    std::vector<Operator> outputs{relu_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.ai_core_spec.cube_vector_split = true;
    opti_compilation_info.soc_version = "Ascend920A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend920A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("Conv2DbpInputDilationFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    fe::PlatformInfoManager::Instance().platform_info_map_.clear();

    bool findNode = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Dilation") {
            findNode = true;
        }
    }
    EXPECT_EQ(findNode, true);
}

TEST_F(conv2dbackprop_dilation_fusion_test, conv2dbackprop_dilation_fusion_test_2) {
    ge::Graph graph("conv2dbackprop_dilation_fusion_test_2");

    auto dedw_shape = vector<int64_t>({512, 512, 3, 3});
    ge::TensorDesc desc_dedw(ge::Shape(dedw_shape), FORMAT_NCHW, DT_FLOAT16);
    auto dedy_shape = vector<int64_t>({2, 512, 14, 14});
    ge::TensorDesc desc_dedy(ge::Shape(dedy_shape), FORMAT_NCHW, DT_FLOAT16);

    auto data_dw = op::Data("data_dw").set_attr_index(0);
    data_dw.update_input_desc_x(desc_dedw);
    data_dw.update_output_desc_y(desc_dedw);

    auto data_dy = op::Data("data_dy").set_attr_index(1);
    data_dy.update_input_desc_x(desc_dedy);
    data_dy.update_output_desc_y(desc_dedy);

    auto conv2dbackpropinputd = op::Conv2DBackpropInputD("conv2dbackpropinputd")
        .set_input_filter(data_dw)
        .set_input_out_backprop(data_dy)
        .set_attr_input_size({2, 512, 28, 28})
        .set_attr_strides({1, 1, 2, 2})
        .set_attr_pads({0, 1, 0, 1})
        .set_attr_dilations({1, 1, 1, 1})
        .set_attr_groups({1})
        .set_attr_data_format("NCHW");
    auto out_shape = ge::Shape({2, 512, 28, 28});
    ge::TensorDesc desc_out(ge::Shape(out_shape), FORMAT_NCHW, DT_FLOAT16);
    conv2dbackpropinputd.update_output_desc_y(desc_out);

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(conv2dbackpropinputd);

    std::vector<Operator> inputs{data_dw, data_dy};
    std::vector<Operator> outputs{relu_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.ai_core_spec.cube_vector_split = true;
    opti_compilation_info.soc_version = "Ascend920A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend920A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("Conv2DbpInputDilationFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    fe::PlatformInfoManager::Instance().platform_info_map_.clear();

    bool findNode = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Dilation") {
            findNode = true;
        }
    }
    EXPECT_EQ(findNode, false);
}

TEST_F(conv2dbackprop_dilation_fusion_test, conv2dbackprop_dilation_fusion_test_3) {
    ge::Graph graph("conv2dbackprop_dilation_fusion_test_3");

    auto dedw_shape = vector<int64_t>({512, 512, 3, 3});
    ge::TensorDesc desc_dedw(ge::Shape(dedw_shape), FORMAT_NCHW, DT_FLOAT16);
    auto dedy_shape = vector<int64_t>({2, 512, 5, 5});
    ge::TensorDesc desc_dedy(ge::Shape(dedy_shape), FORMAT_NCHW, DT_FLOAT16);

    auto data_dw = op::Data("data_dw").set_attr_index(0);
    data_dw.update_input_desc_x(desc_dedw);
    data_dw.update_output_desc_y(desc_dedw);

    auto data_dy = op::Data("data_dy").set_attr_index(1);
    data_dy.update_input_desc_x(desc_dedy);
    data_dy.update_output_desc_y(desc_dedy);

    auto conv2dbackpropinputd = op::Conv2DBackpropInputD("conv2dbackpropinputd")
        .set_input_filter(data_dw)
        .set_input_out_backprop(data_dy)
        .set_attr_input_size({2, 512, 18, 18})
        .set_attr_strides({1, 1, 4, 4})
        .set_attr_pads({1, 1, 1, 1})
        .set_attr_dilations({1, 1, 1, 1})
        .set_attr_groups({1})
        .set_attr_data_format("NCHW");
    auto out_shape = ge::Shape({2, 512, 18, 18});
    ge::TensorDesc desc_out(ge::Shape(out_shape), FORMAT_NCHW, DT_FLOAT16);
    conv2dbackpropinputd.update_output_desc_y(desc_out);

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(conv2dbackpropinputd);

    std::vector<Operator> inputs{data_dw, data_dy};
    std::vector<Operator> outputs{relu_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.ai_core_spec.cube_vector_split = true;
    opti_compilation_info.soc_version = "Ascend920A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend920A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("Conv2DbpInputDilationFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    fe::PlatformInfoManager::Instance().platform_info_map_.clear();

    bool findNode = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Dilation") {
            findNode = true;
        }
    }
    EXPECT_EQ(findNode, true);
}

TEST_F(conv2dbackprop_dilation_fusion_test, conv2dbackprop_dilation_fusion_test_4) {
    ge::Graph graph("conv2dbackprop_dilation_fusion_test_1");

    auto dedw_shape = vector<int64_t>({512, 1, 1, 512});
    ge::TensorDesc desc_dedw(ge::Shape(dedw_shape), FORMAT_NHWC, DT_FLOAT16);
    auto dedy_shape = vector<int64_t>({2, 14, 14, 512});
    ge::TensorDesc desc_dedy(ge::Shape(dedy_shape), FORMAT_NHWC, DT_FLOAT16);

    auto data_dw = op::Data("data_dw").set_attr_index(0);
    data_dw.update_input_desc_x(desc_dedw);
    data_dw.update_output_desc_y(desc_dedw);

    auto data_dy = op::Data("data_dy").set_attr_index(1);
    data_dy.update_input_desc_x(desc_dedy);
    data_dy.update_output_desc_y(desc_dedy);

    auto conv2dbackpropinputd = op::Conv2DBackpropInputD("conv2dbackpropinputd")
        .set_input_filter(data_dw)
        .set_input_out_backprop(data_dy)
        .set_attr_input_size({2,28, 28, 512})
        .set_attr_strides({1, 2, 2,1})
        .set_attr_pads({0, 0, 0, 0})
        .set_attr_dilations({1, 1, 1, 1})
        .set_attr_groups({1})
        .set_attr_data_format("NHWC");
    auto out_shape = ge::Shape({2,28, 28, 512});
    ge::TensorDesc desc_out(ge::Shape(out_shape), FORMAT_NHWC, DT_FLOAT16);
    conv2dbackpropinputd.update_output_desc_y(desc_out);

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(conv2dbackpropinputd);

    std::vector<Operator> inputs{data_dw, data_dy};
    std::vector<Operator> outputs{relu_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.ai_core_spec.cube_vector_split = true;
    opti_compilation_info.soc_version = "Ascend920A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend920A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("Conv2DbpInputDilationFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    fe::PlatformInfoManager::Instance().platform_info_map_.clear();

    bool findNode = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Dilation") {
            findNode = true;
        }
    }
    EXPECT_EQ(findNode, true);
}

TEST_F(conv2dbackprop_dilation_fusion_test, conv2d_transpose_dilation_fusion_test_0) {
    ge::Graph graph("conv2dbackprop_dilation_fusion_test_1");

    auto dedw_shape = vector<int64_t>({512, 1, 1, 512});
    ge::TensorDesc desc_dedw(ge::Shape(dedw_shape), FORMAT_NHWC, DT_FLOAT16);
    auto dedy_shape = vector<int64_t>({2, 14, 14, 512});
    ge::TensorDesc desc_dedy(ge::Shape(dedy_shape), FORMAT_NHWC, DT_FLOAT16);

    auto data_dw = op::Data("data_dw").set_attr_index(0);
    data_dw.update_input_desc_x(desc_dedw);
    data_dw.update_output_desc_y(desc_dedw);

    auto data_dy = op::Data("data_dy").set_attr_index(1);
    data_dy.update_input_desc_x(desc_dedy);
    data_dy.update_output_desc_y(desc_dedy);

    auto conv2d_transpose_d = op::Conv2DTransposeD("Conv2DTransposeD")
        .set_input_x(data_dy)
        .set_input_filter(data_dw)
        .set_attr_input_size({2,28, 28, 512})
        .set_attr_strides({1, 2, 2,1})
        .set_attr_pads({0, 0, 0, 0})
        .set_attr_dilations({1, 1, 1, 1})
        .set_attr_groups({1})
        .set_attr_data_format("NHWC");
    auto out_shape = ge::Shape({2,28, 28, 512});
    ge::TensorDesc desc_out(ge::Shape(out_shape), FORMAT_NHWC, DT_FLOAT16);
    conv2d_transpose_d.update_output_desc_y(desc_out);

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(conv2d_transpose_d);

    std::vector<Operator> inputs{data_dw, data_dy};
    std::vector<Operator> outputs{relu_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.ai_core_spec.cube_vector_split = true;
    opti_compilation_info.soc_version = "Ascend920A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend920A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("Conv2DbpInputDilationFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    fe::PlatformInfoManager::Instance().platform_info_map_.clear();

    bool findNode = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Dilation") {
            findNode = true;
        }
    }
    EXPECT_EQ(findNode, true);
}
