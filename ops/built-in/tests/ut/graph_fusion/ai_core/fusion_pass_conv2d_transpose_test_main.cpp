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

class conv2d_transpose_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "conv2d_transpose_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "conv2d_transpose_fusion_test TearDown" << std::endl;
    }
};

TEST_F(conv2d_transpose_fusion_test, conv2d_transpose_fusion_test_1) {
    ge::Graph graph("conv2d_transpose_fusion_test_1");
    auto shape_data = vector<int64_t>({128, 64, 224, 224});
    TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NCHW, DT_FLOAT16);

    auto data = op::Data("data");
    data.update_input_desc_x(desc_data);
    data.update_output_desc_y(desc_data);
    // const op: conv2dtransposed weight
    auto weight_shape = ge::Shape({64, 3, 3, 3});
    TensorDesc desc_weight_1(weight_shape, FORMAT_NCHW, DT_FLOAT16);
    // TensorDesc desc_weight_1(weight_shape, FORMAT_NCHW, DT_FLOAT16);
    Tensor weight_tensor(desc_weight_1);
    auto conv_weight = op::Const("weight")
        .set_attr_value(weight_tensor);

    auto input_size_shape = ge::Shape({4});
    TensorDesc desc_input_size_1(input_size_shape, FORMAT_ND, DT_INT32);
    // TensorDesc desc_weight_1(weight_shape, FORMAT_NCHW, DT_FLOAT16);
    Tensor input_size_tensor(desc_input_size_1);
    uint32_t *input_size_tensor_value = new uint32_t[4]{128, 3, 226, 226};
    input_size_tensor.SetData((uint8_t *) input_size_tensor_value, 4 * 4);

    auto conv_input_size = op::Const("input_size")
        .set_attr_value(input_size_tensor);
    // conv2dtransposed op
    auto conv2dtranspose = op::Conv2DTranspose("conv2dtranspose");
    conv2dtranspose.set_input_input_size(conv_input_size);
    conv2dtranspose.set_input_x(data);
    conv2dtranspose.set_input_filter(conv_weight);
    conv2dtranspose.set_attr_strides({1, 1, 1, 1});
    conv2dtranspose.set_attr_pads({0, 0, 0, 0});
    conv2dtranspose.set_attr_output_padding({0, 0, 0, 0});
    conv2dtranspose.set_attr_dilations({1, 1, 1, 1});
    conv2dtranspose.set_attr_data_format("NCHW");

    TensorDesc conv2dtranspose_input_desc_x(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
    TensorDesc conv2dtranspose_input_desc_filter(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
    TensorDesc conv2dtranspose_output_desc_y(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
    conv2dtranspose.update_input_desc_x(conv2dtranspose_input_desc_x);
    conv2dtranspose.update_input_desc_filter(conv2dtranspose_input_desc_filter);
    conv2dtranspose.update_output_desc_y(conv2dtranspose_output_desc_y);

    std::vector<Operator> inputs{data};
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(conv2dtranspose);
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("AAConstToAttrPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_after");

    bool findD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2DTransposeD") {
            findD = true;
        }
    }
    EXPECT_EQ(findD, true);
    delete[] input_size_tensor_value;

}
