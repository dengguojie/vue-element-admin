#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class conv3d_bp_filter_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "conv3d_bp_filter_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "conv3d_bp_filter_fusion_test TearDown" << std::endl;
    }
};

TEST_F(conv3d_bp_filter_fusion_test, conv3d_bp_filter_fusion_test_1) {
    ge::Graph graph("conv3d_bp_filter_fusion_test_1");
    auto dedy_shape = vector<int64_t>({4, 6, 28, 28, 224});
    TensorDesc desc_dedy(ge::Shape(dedy_shape), FORMAT_NDHWC, DT_FLOAT16);

    auto data1 = op::Data("data1");
    data1.update_input_desc_x(desc_dedy);
    data1.update_output_desc_y(desc_dedy);

    auto dx_shape = vector<int64_t>({4, 6, 56, 56, 96});
    TensorDesc desc_dx(ge::Shape(dx_shape), FORMAT_NDHWC, DT_FLOAT16);

    auto data2 = op::Data("data2")
      .set_attr_index(1);
    data2.update_input_desc_x(desc_dx);
    data2.update_output_desc_y(desc_dx);

    // conv3dtransposed op
    auto conv3dbackpropfilterd = op::Conv3DBackpropFilterD("Conv3DBackpropFilterD")
      .set_input_x(data2)
      .set_input_out_backprop(data1)
      .set_attr_filter_size({1, 1, 5, 3, 224})
      .set_attr_strides({1, 1, 2, 2, 1})
      .set_attr_pads({0, 0, 0, 0, 1, 2})
      .set_attr_dilations({1, 1, 1, 1, 1})
      .set_attr_groups(32)
      .set_attr_data_format("NDHWC");

    TensorDesc conv3dbackpropfilterd_input_desc_out_backprop(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    TensorDesc conv3dbackpropfilterd_input_desc_x(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    TensorDesc conv3dbackpropfilterd_output_desc_y(ge::Shape(), FORMAT_DHWCN, DT_FLOAT);
    conv3dbackpropfilterd.update_input_desc_out_backprop(conv3dbackpropfilterd_input_desc_out_backprop);
    conv3dbackpropfilterd.update_input_desc_x(conv3dbackpropfilterd_input_desc_x);
    conv3dbackpropfilterd.update_output_desc_y(conv3dbackpropfilterd_output_desc_y);

    std::vector<Operator> inputs{data1, data2};
    std::vector<Operator> outputs{conv3dbackpropfilterd};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("Conv3DBpFilterGroupFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv3DBackpropFilterD") {
            findD = true;
        }
    }
    EXPECT_EQ(findD, true);
}
