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

class conv2dbackprop_filter_mul_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "conv2dbackprop_filter_mul_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "conv2dbackprop_filter_mul_fusion_test TearDown" << std::endl;
    }
};

TEST_F(conv2dbackprop_filter_mul_fusion_test, conv2dbackprop_filter_mul_fusion_test_1) {
    ge::Graph graph("conv2dbackprop_filter_mul_fusion_test_1");

    auto dedy_shape = vector<int64_t>({1, 128, 214, 214});
    ge::TensorDesc desc_dedy(ge::Shape(dedy_shape), FORMAT_NCHW, DT_FLOAT16);
    auto data1 = op::Data("data1");
    data1.update_input_desc_x(desc_dedy);
    data1.update_output_desc_y(desc_dedy);

    auto dedx_shape = vector<int64_t>({1, 64, 214, 214});
    ge::TensorDesc desc_dedx(ge::Shape(dedx_shape), FORMAT_NCHW, DT_FLOAT16);
    auto data2 = op::Data("data2").set_attr_index(1);
    data2.update_input_desc_x(desc_dedx);
    data2.update_output_desc_y(desc_dedx);

    auto conv2dbackpropfilterd = op::Conv2DBackpropFilterD("conv2dbackpropfilterd")
        .set_input_x(data2)
        .set_input_out_backprop(data1)
        .set_attr_filter_size({128,16,1,1})
        .set_attr_strides({1,1,1,1})
        .set_attr_pads({0,0,0,0})
        .set_attr_dilations({1,1,1,1})
        .set_attr_groups({4})
        .set_attr_data_format("NCHW");
    ge::TensorDesc input_desc_outbackprop(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
    ge::TensorDesc input_desc_x(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
    ge::TensorDesc output_desc_y(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
    conv2dbackpropfilterd.update_input_desc_out_backprop(input_desc_outbackprop);
    conv2dbackpropfilterd.update_input_desc_x(input_desc_x);
    conv2dbackpropfilterd.update_output_desc_y(output_desc_y);

    std::vector<Operator> inputs{data1, data2};
    std::vector<Operator> outputs{conv2dbackpropfilterd};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("Conv2DbpFilterMulFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMul = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            findMul = true;
        }
    }
    EXPECT_EQ(findMul, true);
}