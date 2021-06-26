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

class depthwise_dw_mul_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "depthwise_dw_mul_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "depthwise_dw_mul_fusion_test TearDown" << std::endl;
    }
};

TEST_F(depthwise_dw_mul_fusion_test, depthwise_dw_mul_fusion_test_1) {
    ge::Graph graph("depthwise_dw_mul_fusion_test_1");

    auto dedy_shape = vector<int64_t>({1, 128, 214, 214});
    ge::TensorDesc desc_dedy(ge::Shape(dedy_shape), FORMAT_NCHW, DT_FLOAT16);
    auto data1 = op::Data("data1");
    data1.update_input_desc_x(desc_dedy);
    data1.update_output_desc_y(desc_dedy);

    auto dedx_shape = vector<int64_t>({1, 128, 214, 214});
    ge::TensorDesc desc_dedx(ge::Shape(dedx_shape), FORMAT_NCHW, DT_FLOAT16);
    auto data2 = op::Data("data2").set_attr_index(1);
    data2.update_input_desc_x(desc_dedx);
    data2.update_output_desc_y(desc_dedx);

    auto filter_shape = vector<int64_t>({4});
    ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_ND, DT_INT32);
    auto data3 = op::Data("data3").set_attr_index(1);
    data3.update_input_desc_x(desc_filter);
    data3.update_output_desc_y(desc_filter);

    auto depthwiseConv2DBackpropFilter = op::DepthwiseConv2DBackpropFilter("DepthwiseConv2DBackpropFilter")
        .set_input_input(data2)
        .set_input_out_backprop(data1)
        .set_input_filter_size(data3)
        .set_attr_strides({1,1,1,1})
        .set_attr_pads({0,0,0,0})
        .set_attr_dilations({1,1,1,1})
        .set_attr_data_format("NCHW");
    depthwiseConv2DBackpropFilter.SetAttr("_fuzz_build", true);
    ge::TensorDesc input_desc_out_backprop(ge::Shape({1, 128, 214, 214}), FORMAT_NCHW, DT_FLOAT16);
    ge::TensorDesc input_desc_input(ge::Shape({1, 128, 214, 214}), FORMAT_NCHW, DT_FLOAT16);
    ge::TensorDesc input_desc_filter_size(ge::Shape({4}), FORMAT_ND, DT_INT32);
    ge::TensorDesc output_desc_filter_grad(ge::Shape({128,16,1,1}), FORMAT_NCHW, DT_FLOAT);
    depthwiseConv2DBackpropFilter.update_input_desc_out_backprop(input_desc_out_backprop);
    depthwiseConv2DBackpropFilter.update_input_desc_input(input_desc_input);
    depthwiseConv2DBackpropFilter.update_input_desc_filter_size(input_desc_filter_size);
    depthwiseConv2DBackpropFilter.update_output_desc_filter_grad(output_desc_filter_grad);

    std::vector<Operator> inputs{data1, data2, data3};
    std::vector<Operator> outputs{depthwiseConv2DBackpropFilter};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("DepthwiseDwMulFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMul = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            findMul = true;
        }
    }
    EXPECT_EQ(findMul, true);
}
