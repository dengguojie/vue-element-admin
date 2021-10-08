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

using namespace ge;
using namespace op;

class depthwise_df_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "depthwise_df_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "depthwise_df_fusion_test TearDown" << std::endl;
    }
};

TEST_F(depthwise_df_fusion_test, depthwise_df_fusion_test_dynamic_HWCN) {
    ge::Graph graph("depthwise_df_fusion_test_dynamic_HWCN");

    auto filter_shape = vector<int64_t>({5, 5, 16, 1});
    ge::TensorDesc desc_filter(ge::Shape(filter_shape), ge::FORMAT_HWCN, ge::DT_FLOAT16);
    auto data_filter = op::Data("data_filter");
    data_filter.update_input_desc_x(desc_filter);
    data_filter.update_output_desc_y(desc_filter);

    auto dedy_shape = vector<int64_t>({2, -1, 214, 16});
    ge::TensorDesc desc_dedy(ge::Shape(dedy_shape), FORMAT_NHWC, DT_FLOAT16);
    auto data_dedy = op::Data("data_dedy");
    data_dedy.update_input_desc_x(desc_dedy);
    data_dedy.update_output_desc_y(desc_dedy);

    auto dedx_shape = vector<int64_t>({2, -1, 214, 16});
    auto fmap_shape = vector<int64_t>({4});
    ge::TensorDesc desc_fmap(ge::Shape(fmap_shape), FORMAT_ND, DT_INT32);
    auto data_fmap = op::Data("data_fmap").set_attr_index(0);
    data_fmap.update_input_desc_x(desc_fmap);
    data_fmap.update_output_desc_y(desc_fmap);

    auto depthwiseConv2DBackpropInput = op::DepthwiseConv2DBackpropInput("DepthwiseConv2DBackpropInput")
        .set_input_input_size(data_fmap)
        .set_input_filter(data_filter)
        .set_input_out_backprop(data_dedy)
        .set_attr_strides({1,1,1,1})
        .set_attr_pads({2,2,2,2})
        .set_attr_dilations({1,1,1,1})
        .set_attr_data_format("NHWC");
    ge::TensorDesc input_desc_input_size(ge::Shape(fmap_shape), FORMAT_ND, DT_INT32);
    ge::TensorDesc input_desc_filter(ge::Shape(filter_shape), FORMAT_HWCN, DT_FLOAT16);
    ge::TensorDesc input_desc_out_backprop(ge::Shape(dedy_shape), FORMAT_NHWC, DT_FLOAT16);    
    ge::TensorDesc output_desc_input_grad(ge::Shape(dedx_shape), FORMAT_NHWC, DT_FLOAT);
    depthwiseConv2DBackpropInput.update_input_desc_input_size(input_desc_input_size);
    depthwiseConv2DBackpropInput.update_input_desc_filter(input_desc_filter);
    depthwiseConv2DBackpropInput.update_input_desc_out_backprop(input_desc_out_backprop);
    depthwiseConv2DBackpropInput.update_output_desc_input_grad(output_desc_input_grad);

    std::vector<Operator> inputs{data_fmap, data_filter, data_dedy};
    std::vector<Operator> outputs{depthwiseConv2DBackpropInput};
    graph.SetInputs(inputs).SetOutputs(outputs);

    vector<int64_t> expect_filter_shape = vector<int64_t>({5, 5, 1, 16});
    vector<int64_t> actual_filter_shape = vector<int64_t>({5, 5, 16, 1});
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("DepthwiseDfFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DepthwiseConv2DBackpropInput") {
            auto depthwise_conv2dbp_input_node_desc = node->GetOpDesc();
            auto filter_tensor = depthwise_conv2dbp_input_node_desc->GetInputDesc(1);
            actual_filter_shape = filter_tensor.GetShape().GetDims();
            break;
        }
    }
    EXPECT_EQ(actual_filter_shape, expect_filter_shape);
}

TEST_F(depthwise_df_fusion_test, depthwise_df_fusion_test_dynamic_NCHW) {
    ge::Graph graph("depthwise_df_fusion_test_dynamic_NCHW");

    auto filter_shape = vector<int64_t>({16, 1, 5, 5});
    ge::TensorDesc desc_filter(ge::Shape(filter_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    auto data_filter = op::Data("data_filter");
    data_filter.update_input_desc_x(desc_filter);
    data_filter.update_output_desc_y(desc_filter);

    auto dedy_shape = vector<int64_t>({2, 16, -1, 214});
    ge::TensorDesc desc_dedy(ge::Shape(dedy_shape), FORMAT_NCHW, DT_FLOAT16);
    auto data_dedy = op::Data("data_dedy");
    data_dedy.update_input_desc_x(desc_dedy);
    data_dedy.update_output_desc_y(desc_dedy);

    auto dedx_shape = vector<int64_t>({2, 16, -1, 214});
    auto fmap_shape = vector<int64_t>({4});
    ge::TensorDesc desc_fmap(ge::Shape(fmap_shape), FORMAT_ND, DT_INT32);
    auto data_fmap = op::Data("data_fmap").set_attr_index(0);
    data_fmap.update_input_desc_x(desc_fmap);
    data_fmap.update_output_desc_y(desc_fmap);

    auto depthwiseConv2DBackpropInput = op::DepthwiseConv2DBackpropInput("DepthwiseConv2DBackpropInput")
        .set_input_input_size(data_fmap)
        .set_input_filter(data_filter)
        .set_input_out_backprop(data_dedy)
        .set_attr_strides({1,1,1,1})
        .set_attr_pads({2,2,2,2})
        .set_attr_dilations({1,1,1,1})
        .set_attr_data_format("NCHW");
    ge::TensorDesc input_desc_input_size(ge::Shape(fmap_shape), FORMAT_ND, DT_INT32);
    ge::TensorDesc input_desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
    ge::TensorDesc input_desc_out_backprop(ge::Shape(dedy_shape), FORMAT_NCHW, DT_FLOAT16);    
    ge::TensorDesc output_desc_input_grad(ge::Shape(dedx_shape), FORMAT_NCHW, DT_FLOAT);
    depthwiseConv2DBackpropInput.update_input_desc_input_size(input_desc_input_size);
    depthwiseConv2DBackpropInput.update_input_desc_filter(input_desc_filter);
    depthwiseConv2DBackpropInput.update_input_desc_out_backprop(input_desc_out_backprop);
    depthwiseConv2DBackpropInput.update_output_desc_input_grad(output_desc_input_grad);

    std::vector<Operator> inputs{data_fmap, data_filter, data_dedy};
    std::vector<Operator> outputs{depthwiseConv2DBackpropInput};
    graph.SetInputs(inputs).SetOutputs(outputs);

    vector<int64_t> expect_filter_shape = vector<int64_t>({16, 1, 5, 5});
    vector<int64_t> actual_filter_shape = vector<int64_t>({16, 1, 5, 5});
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("DepthwiseDfFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DepthwiseConv2DBackpropInput") {
            auto depthwise_conv2dbp_input_node_desc = node->GetOpDesc();
            auto filter_tensor = depthwise_conv2dbp_input_node_desc->GetInputDesc(1);
            actual_filter_shape = filter_tensor.GetShape().GetDims();
            break;
        }
    }
    EXPECT_EQ(actual_filter_shape, expect_filter_shape);
}