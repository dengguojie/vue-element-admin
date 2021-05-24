#include "gtest/gtest.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "nn_calculation_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class conv3d_backprop_input_bias_add_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "conv3d_backprop_input_bias_add_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "conv3d_backprop_input_bias_add_fusion_test TearDown" << std::endl;
    }
};

TEST_F(conv3d_backprop_input_bias_add_fusion_test, conv3d_backprop_input_bias_add_fusion_base_case1) {
    ge::Graph graph("conv3d_backprop_input_bias_add_fusion_base_case1");

    auto filter_shape = vector<int64_t>({2,2,2,320,320});
    ge::TensorDesc filter_desc(ge::Shape(filter_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);
    auto data_filter = op::Data("data_filter");
    data_filter.update_input_desc_x(filter_desc);
    data_filter.update_output_desc_y(filter_desc);

    auto out_backprop_shape = vector<int64_t>({4,4,4,4,320});
    ge::TensorDesc out_backprop_desc(ge::Shape(out_backprop_shape), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    auto data_out_backprop = op::Data("data_out_backprop");
    data_out_backprop.update_input_desc_x(out_backprop_desc);
    data_out_backprop.update_output_desc_y(out_backprop_desc);

    auto bias_shape = vector<int64_t>({320});
    ge::TensorDesc bias_desc(ge::Shape(bias_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);
    auto data_bias = op::Data("data_bias");
    data_bias.update_input_desc_x(bias_desc);
    data_bias.update_output_desc_y(bias_desc);

    auto conv3d_bp_input_d = op::Conv3DBackpropInputD("conv3d_bp_input_d")
        .set_input_filter(data_filter)
        .set_input_out_backprop(data_out_backprop)
        .set_attr_input_size({4, 8, 8, 8, 320})
        .set_attr_strides({1, 2, 2, 2, 1})
        .set_attr_pads({0, 0, 0, 0, 0, 0})
        .set_attr_dilations({1, 1, 1, 1, 1})
        .set_attr_groups({1})
        .set_attr_data_format("NDHWC");

    auto bias_add = op::BiasAdd("bias_add")
        .set_input_x(conv3d_bp_input_d)
        .set_input_bias(data_bias)
        .set_attr_data_format("NDHWC");

    std::vector<Operator> inputs{data_filter, data_out_backprop, data_bias};
    std::vector<Operator> outputs{bias_add};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("Conv3DbpInputBiasAddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool find_transpose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv3DTransposeD") {
            find_transpose = true;
        }
    }
    EXPECT_EQ(find_transpose, true);
}
